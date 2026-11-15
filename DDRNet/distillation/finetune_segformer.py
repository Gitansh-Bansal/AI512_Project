# ------------------------------------------------------------------------------
# 5-Epoch NightCity Adaptation for Teacher SegFormer-B2
# Adapts Cityscapes-pretrained SegFormer-B2 to nighttime illumination:
#   - Learns nighttime sky (pitch black -> sky instead of building)
#   - Adapts to headlights, specular glare, and low-contrast shadows
# Produces high-fidelity soft targets for distilling into PIDNet-S
# ------------------------------------------------------------------------------

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.nightcity import NightCity
from metrics import SegmentationMetrics, CITYSCAPES_CLASSES


def get_segformer_param_groups(model: nn.Module, base_lr: float = 6e-5, encoder_lr_mult: float = 0.33):
    """Differential learning rate: lower LR for pretrained encoder, higher for decoder."""
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "decode_head" in name:
            decoder_params.append(param)
        else:
            encoder_params.append(param)
    return [
        {"params": encoder_params, "lr": base_lr * encoder_lr_mult, "weight_decay": 1e-2},
        {"params": decoder_params, "lr": base_lr, "weight_decay": 1e-2},
    ]


def adjust_learning_rate(optimizer, base_lrs, current_iter, max_iters, power=0.9):
    factor = (1.0 - float(current_iter) / max_iters) ** power
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = base_lrs[i] * factor
    return optimizer.param_groups[1]["lr"]


def train_epoch(model, train_loader, optimizer, device, epoch, max_epochs, base_lrs, global_iter, max_iters, scaler=None):
    model.train()
    total_loss_accum = 0.0
    num_batches = len(train_loader)
    use_amp = (scaler is not None and device.type == "cuda")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Adapt SegFormer]", leave=False)
    for batch_idx, (images, labels, _, _, _) in enumerate(pbar):
        current_iter = epoch * num_batches + batch_idx
        current_lr = adjust_learning_rate(optimizer, base_lrs, current_iter, max_iters)

        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            logits = outputs.logits
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, labels, ignore_index=255)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss_accum += loss.detach()

        if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == num_batches:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

    avg_loss = float((total_loss_accum / max(1, num_batches)).item())
    return avg_loss


def validate_segformer(model, val_loader, device, num_samples: Optional[int] = 50):
    model.eval()
    metric_engine = SegmentationMetrics(num_classes=19, ignore_label=255, class_names=CITYSCAPES_CLASSES)
    eval_count = 0
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for images, labels, _, _, _ in tqdm(val_loader, desc="Validating SegFormer", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(images)
                logits = out.logits
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

            pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
            targets = labels.numpy()
            metric_engine.update(pred_classes, targets)

            eval_count += images.size(0)
            if num_samples and eval_count >= num_samples:
                break

    return metric_engine.compute_metrics()


def run_segformer_adaptation(
    data_root: str = "AI512_Project/Data",
    train_list: str = "PIDNet/data/list/nightcity/train.lst",
    val_list: str = "PIDNet/data/list/nightcity/val.lst",
    epochs: int = 5,
    batch_size: int = 4,
    base_lr: float = 6e-5,
    crop_h: int = 512,
    crop_w: int = 512,
    val_samples: Optional[int] = 50,
    num_train_samples: Optional[int] = None,
    save_dir: str = "output/nightcity_segformer_adapted",
) -> List[Dict]:
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("SEGFORMER-B2 ADAPTATION ON NIGHTCITY (5 EPOCHS)")
    print("=" * 70)
    print(f"[INFO] Compute Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | Base LR: {base_lr}")

    # Datasets
    train_dataset = NightCity(
        root=data_root, list_path=train_list, num_classes=19,
        multi_scale=True, flip=True, crop_size=(crop_h, crop_w)
    )
    val_dataset = NightCity(
        root=data_root, list_path=val_list, num_classes=19,
        multi_scale=False, flip=False, crop_size=(512, 1024)
    )

    if num_train_samples and num_train_samples < len(train_dataset):
        train_dataset.files = train_dataset.files[:num_train_samples]

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)

    num_workers = min(2, os.cpu_count() or 2) if device.type == "cuda" else 0
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4 if device.type == "cuda" else 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"[INFO] Train Samples: {len(train_dataset):,} | Val Samples: {len(val_dataset):,} | Workers: {num_workers}")

    # Load Cityscapes Pretrained SegFormer-B2
    print("[INFO] Loading SegFormer-B2 (nvidia/segformer-b2-finetuned-cityscapes-1024-1024)...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    ).to(device)

    # Initial zero-shot evaluation on NightCity val
    print("[INFO] Evaluating zero-shot SegFormer-B2 baseline on NightCity val...")
    zero_shot_metrics = validate_segformer(model, val_loader, device, num_samples=val_samples)
    print(f"  -> Zero-shot mIoU: {zero_shot_metrics['mIoU']*100.0:.2f}% | Pixel Acc: {zero_shot_metrics['Pixel_Accuracy']*100.0:.2f}%")

    param_groups = get_segformer_param_groups(model, base_lr=base_lr, encoder_lr_mult=0.33)
    base_lrs = [g["lr"] for g in param_groups]
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    max_iters = epochs * len(train_loader)

    history = [{
        "epoch": 0,
        "train_loss": None,
        "val_mIoU": zero_shot_metrics["mIoU"],
        "val_pixel_acc": zero_shot_metrics["Pixel_Accuracy"],
        "time_sec": 0.0,
    }]
    best_miou = zero_shot_metrics["mIoU"]
    best_path = os.path.join(save_dir, "best_segformer_nightcity.pt")

    for epoch in range(epochs):
        t0 = time.time()
        loss = train_epoch(
            model, train_loader, optimizer, device, epoch, epochs,
            base_lrs, epoch * len(train_loader), max_iters, scaler=scaler
        )
        elapsed = time.time() - t0

        metrics = validate_segformer(model, val_loader, device, num_samples=val_samples)
        val_miou = metrics["mIoU"]
        val_acc = metrics["Pixel_Accuracy"]
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss:.4f} | Val mIoU: {val_miou*100.0:.2f}% | Pixel Acc: {val_acc*100.0:.2f}% | Time: {elapsed:.1f}s")

        record = {
            "epoch": epoch + 1,
            "train_loss": float(loss),
            "val_mIoU": float(val_miou),
            "val_pixel_acc": float(val_acc),
            "time_sec": float(elapsed),
        }
        history.append(record)

        if val_miou > best_miou or (epoch == 0):
            best_miou = val_miou
            torch.save(model.state_dict(), best_path)
            print(f"  [CHECKPOINT] New best adapted teacher saved to: {best_path}")

    # Save training history
    history_file = os.path.join(save_dir, "adaptation_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] SegFormer adaptation complete. Best mIoU: {best_miou*100.0:.2f}%. Saved to: {best_path}")
    return history
