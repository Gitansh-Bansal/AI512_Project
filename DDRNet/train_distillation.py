# ------------------------------------------------------------------------------
# Task 8: Knowledge Distillation Training Engine for DDRNet-23-slim
# Student: DDRNet-23-slim (5.68M params, Real-time ~102 FPS)
# Teacher: SegFormer-B2 (27.5M params, Transformer, Pretrained & Fine-tuned on NightCity)
# ------------------------------------------------------------------------------

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR / "segmentation") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "segmentation"))

from DDRNet_23_slim import DualResNet, BasicBlock
from distillation.teacher_segformer import get_segformer_teacher
from distillation.distill_loss import LogitKDLoss, DDRNetDistillationWrapper
from datasets.nightcity import NightCity
from metrics import SegmentationMetrics, CITYSCAPES_CLASSES


class OhemCrossEntropy(nn.Module):
    """
    Online Hard Example Mining Cross Entropy Loss for semantic segmentation.
    Keeps at least min_kept hardest pixels and thresholded pixels.
    """
    def __init__(self, ignore_label: int = 255, thres: float = 0.7, min_kept: int = 50000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thres)
        self.min_kept = int(min_kept)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def forward(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(score, target)
        loss = loss.view(-1)
        valid_mask = target.view(-1) != self.ignore_label
        loss = loss[valid_mask]
        if loss.numel() == 0:
            return score.sum() * 0.0

        if self.min_kept > 0 and loss.numel() > self.min_kept:
            # Use torch.kthvalue instead of torch.sort to prevent GPU stall
            k = loss.numel() - self.min_kept
            threshold_val = torch.kthvalue(loss, k).values
            if threshold_val > self.thresh:
                loss = loss[loss >= self.thresh]
            else:
                loss = loss[loss >= threshold_val]
        return loss.mean()


def adjust_learning_rate(optimizer, base_lr, current_iter, max_iters, power=0.9):
    lr = base_lr * ((1 - float(current_iter) / max_iters) ** power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_epoch(wrapper, train_loader, optimizer, device, epoch, max_epochs, base_lr, global_iter, max_iters, scaler=None):
    wrapper.student.train()
    total_loss_accum = 0.0
    sem_loss_accum = 0.0
    kd_loss_accum = 0.0
    num_batches = len(train_loader)
    use_amp = (scaler is not None and device.type == 'cuda')

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [DDRNet Distill]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        current_iter = epoch * num_batches + batch_idx
        lr = adjust_learning_rate(optimizer, base_lr, current_iter, max_iters)

        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, loss_sem, loss_kd, _ = wrapper(images, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss_accum += loss.detach()
        sem_loss_accum += loss_sem.detach()
        kd_loss_accum += loss_kd.detach()

        if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == num_batches:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sem': f"{loss_sem.item():.4f}",
                'kd': f"{loss_kd.item():.4f}",
                'lr': f"{lr:.6f}"
            })

    avg_total = float((total_loss_accum / max(1, num_batches)).item())
    avg_sem = float((sem_loss_accum / max(1, num_batches)).item())
    avg_kd = float((kd_loss_accum / max(1, num_batches)).item())
    return avg_total, avg_sem, avg_kd


def validate_student(student, val_loader, device, num_samples=None):
    student.eval()
    metric_engine = SegmentationMetrics(num_classes=19, ignore_label=255, class_names=CITYSCAPES_CLASSES)
    eval_count = 0
    use_amp = (device.type == 'cuda')

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating DDRNet Student", leave=False):
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            target = labels.numpy()

            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = student(images)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                if preds.shape[-2:] != labels.shape[-2:]:
                    preds = F.interpolate(preds, size=labels.shape[-2:], mode="bilinear", align_corners=True)

            pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
            metric_engine.update(pred_labels, target)

            eval_count += images.size(0)
            if num_samples is not None and eval_count >= num_samples:
                break

    return metric_engine.compute_metrics()


def run_training_distillation(cfg: Dict) -> Dict:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Starting DDRNet-23-slim Distillation on {device} ===")

    epochs = cfg.get("epochs", 30)
    batch_size = cfg.get("batch_size", 4 if device.type == "cuda" else 2)
    base_lr = cfg.get("base_lr", 0.001)
    kd_weight = cfg.get("kd_weight", 0.2)
    kd_temp = cfg.get("kd_temperature", 2.0)
    aux_weight = cfg.get("aux_weight", 0.4)
    save_dir = Path(cfg.get("save_dir", "output/nightcity_ddrnet_distilled"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Loaders
    data_root = cfg["data_root"]
    train_list = cfg.get("train_list", None)
    val_list = cfg.get("val_list", None)

    train_ds = NightCity(
        root=data_root,
        list_path=train_list,
        num_classes=19,
        multi_scale=False,
        flip=True,
        crop_size=cfg.get("crop_size", (512, 1024)),
    )
    if cfg.get("max_train_samples"):
        train_ds.files = train_ds.files[:cfg["max_train_samples"]]

    val_ds = NightCity(
        root=data_root,
        list_path=val_list,
        num_classes=19,
        multi_scale=False,
        flip=False,
        crop_size=(512, 1024),
    )

    num_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    print(f"Dataset: Train samples={len(train_ds)} | Val samples={len(val_ds)}")

    # 2. Instantiate Student DDRNet-23-slim
    student = DualResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=19,
        planes=32,
        spp_planes=128,
        head_planes=64,
        augment=True
    ).to(device)

    # 3. Instantiate Teacher SegFormer-B2
    teacher_weights = cfg.get("teacher_weights", None)
    teacher = get_segformer_teacher(device, weights_path=teacher_weights)

    # 4. Losses & Wrapper
    sem_loss = OhemCrossEntropy(ignore_label=255, thres=0.7, min_kept=50000)
    kd_loss = LogitKDLoss(temperature=kd_temp, ignore_label=255)
    wrapper = DDRNetDistillationWrapper(
        student=student,
        teacher=teacher,
        sem_loss=sem_loss,
        kd_loss=kd_loss,
        aux_weight=aux_weight,
        kd_weight=kd_weight
    ).to(device)

    # 5. Optimizer
    opt_type = cfg.get("optimizer", "adamw").lower()
    if opt_type == "sgd":
        optimizer = torch.optim.SGD(student.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=base_lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    max_iters = epochs * len(train_loader)

    history = {
        "train_loss": [],
        "sem_loss": [],
        "kd_loss": [],
        "val_mIoU": [],
        "val_pixel_acc": [],
        "epoch_times": []
    }

    best_mIoU = -1.0
    val_freq = cfg.get("val_freq", 2)
    val_samples = cfg.get("val_samples", None)

    for epoch in range(epochs):
        t0 = time.time()
        loss_tot, loss_sem, loss_kd = train_epoch(
            wrapper, train_loader, optimizer, device, epoch, epochs, base_lr,
            epoch * len(train_loader), max_iters, scaler=scaler
        )
        elapsed = time.time() - t0

        history["train_loss"].append(loss_tot)
        history["sem_loss"].append(loss_sem)
        history["kd_loss"].append(loss_kd)
        history["epoch_times"].append(elapsed)

        log_str = f"[Epoch {epoch+1}/{epochs}] Loss: {loss_tot:.4f} (Sem: {loss_sem:.4f}, KD: {loss_kd:.4f}) | Time: {elapsed:.1f}s"
        print(log_str)

        if (epoch + 1) % val_freq == 0 or (epoch + 1) == epochs:
            val_metrics = validate_student(student, val_loader, device, num_samples=val_samples)
            curr_mIoU = val_metrics["mIoU"] * 100.0
            curr_acc = val_metrics["Pixel_Accuracy"] * 100.0
            history["val_mIoU"].append(curr_mIoU)
            history["val_pixel_acc"].append(curr_acc)

            print(f"  -> Student Validation mIoU: {curr_mIoU:.2f}% | Pixel Acc: {curr_acc:.2f}%")

            # Save latest checkpoint
            torch.save({
                "epoch": epoch + 1,
                "state_dict": student.state_dict(),
                "mIoU": curr_mIoU,
                "history": history
            }, save_dir / "latest_ddrnet_distilled.pt")

            # Save best checkpoint
            if curr_mIoU > best_mIoU:
                best_mIoU = curr_mIoU
                torch.save({
                    "epoch": epoch + 1,
                    "state_dict": student.state_dict(),
                    "mIoU": best_mIoU,
                    "pixel_acc": curr_acc,
                    "history": history
                }, save_dir / "best_ddrnet_distilled.pt")
                print(f"  [CHECKPOINT] New best student mIoU ({best_mIoU:.2f}%) saved to {save_dir / 'best_ddrnet_distilled.pt'}")

    with open(save_dir / "distill_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def plot_distillation_training_history(history: Dict, save_path: Optional[str] = None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Total Distill Loss", color="red", lw=2)
    ax1.plot(epochs, history["sem_loss"], label="Semantic CrossEntropy", color="blue", ls="--")
    ax1.plot(epochs, history["kd_loss"], label="Logit KD Loss", color="green", ls=":")
    ax1.set_title("DDRNet-23-slim Distillation Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if history.get("val_mIoU"):
        val_x = np.linspace(1, len(history["train_loss"]), len(history["val_mIoU"]))
        ax2.plot(val_x, history["val_mIoU"], label="mIoU (%)", color="magenta", marker="o", lw=2)
        if history.get("val_pixel_acc"):
            ax2.plot(val_x, history["val_pixel_acc"], label="Pixel Acc (%)", color="orange", marker="s", lw=2)
        ax2.set_title("Student NightCity Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Plot saved to {save_path}")
    plt.show()
