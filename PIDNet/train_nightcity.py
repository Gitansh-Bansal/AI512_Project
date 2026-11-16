# ------------------------------------------------------------------------------
# Task 3: PIDNet-S Training Engine on NightCity Dataset
# Implements composite loss: OHEM / CE on Detail and Main branches + Boundary BCE
# Optimized with AMP FP16, non-blocking tensors, kthvalue OHEM, and pinned memory.
# Supports CPU and GPU execution with checkpointing and validation tracking.
# ------------------------------------------------------------------------------

import os
import sys
import time
import json
import argparse
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

from models.pidnet import PIDNet
from datasets.nightcity import NightCity
from metrics import SegmentationMetrics, CITYSCAPES_CLASSES
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss


class FullModelWrapper(nn.Module):
    """
    Wraps PIDNet-S with semantic segmentation loss (Detail + Main heads)
    and Boundary BCE loss for the Derivative branch.
    """
    def __init__(self, model: nn.Module, sem_loss: nn.Module, bd_loss: nn.Module, bd_weight: float = 1.0):
        super(FullModelWrapper, self).__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss
        self.bd_weight = bd_weight

    def forward(self, inputs, labels, bd_gt):
        # outputs is [x_extra_p, x_, x_extra_d] when model.augment is True
        outputs = self.model(inputs)
        h, w = labels.size(1), labels.size(2)

        # Upsample all branch outputs to ground truth resolution
        for i in range(len(outputs)):
            if outputs[i].size(2) != h or outputs[i].size(3) != w:
                outputs[i] = F.interpolate(
                    outputs[i], size=(h, w), mode='bilinear', align_corners=False
                )

        # Semantic loss on Detail (outputs[0]) and Main (outputs[1]) heads
        loss_sem = self.sem_loss(outputs[:-1], labels).mean()
        # Boundary loss on Derivative head (outputs[2])
        loss_bd = self.bd_loss(outputs[-1], bd_gt).mean()

        total_loss = loss_sem + self.bd_weight * loss_bd
        return total_loss, loss_sem, loss_bd, outputs[1]


def adjust_learning_rate(optimizer, base_lr, current_iter, max_iters, power=0.9):
    """Polynomial learning rate decay."""
    lr = base_lr * ((1 - float(current_iter) / max_iters) ** power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_epoch(wrapper, train_loader, optimizer, device, epoch, max_epochs, base_lr, global_iter, max_iters, scaler=None):
    wrapper.train()
    total_loss_accum = 0.0
    sem_loss_accum = 0.0
    bd_loss_accum = 0.0
    num_batches = len(train_loader)
    use_amp = (scaler is not None and device.type == 'cuda')

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train Baseline]", leave=False)
    for batch_idx, (images, labels, bd_gts, _, _) in enumerate(pbar):
        current_iter = epoch * num_batches + batch_idx
        lr = adjust_learning_rate(optimizer, base_lr, current_iter, max_iters)

        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        bd_gts = bd_gts.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, loss_sem, loss_bd, _ = wrapper(images, labels, bd_gts)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss_accum += loss.detach()
        sem_loss_accum += loss_sem.detach()
        bd_loss_accum += loss_bd.detach()

        if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == num_batches:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sem': f"{loss_sem.item():.4f}",
                'bd': f"{loss_bd.item():.4f}",
                'lr': f"{lr:.6f}"
            })

    avg_total = float((total_loss_accum / max(1, num_batches)).item())
    avg_sem = float((sem_loss_accum / max(1, num_batches)).item())
    avg_bd = float((bd_loss_accum / max(1, num_batches)).item())
    return avg_total, avg_sem, avg_bd


def validate_model(model, val_loader, device, num_samples=None):
    model.eval()
    metric_engine = SegmentationMetrics(num_classes=19, ignore_label=255, class_names=CITYSCAPES_CLASSES)
    eval_count = 0
    use_amp = (device.type == 'cuda')

    with torch.no_grad():
        for images, labels, _, _, _ in tqdm(val_loader, desc="Validating Baseline", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(images)
                if isinstance(preds, (list, tuple)):
                    preds = preds[1]
                if preds.shape[-2:] != labels.shape[-2:]:
                    preds = F.interpolate(preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)

            pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
            target = labels.numpy()
            metric_engine.update(pred_labels, target)

            eval_count += images.size(0)
            if num_samples is not None and eval_count >= num_samples:
                break

    return metric_engine.compute_metrics()


def plot_training_history(history: List[Dict], save_path: Optional[str] = None):
    epochs = [r['epoch'] for r in history]
    train_losses = [r['train_loss_total'] for r in history]
    sem_losses = [r.get('train_loss_sem', 0) for r in history]
    bd_losses = [r.get('train_loss_bd', 0) for r in history]

    val_epochs = [r['epoch'] for r in history if 'val_mIoU' in r]
    val_mIoUs = [r['val_mIoU'] for r in history if 'val_mIoU' in r]
    val_accs = [r.get('val_acc', 0) for r in history if 'val_acc' in r]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Total Loss', linewidth=2)
    axes[0].plot(epochs, sem_losses, 'g--', label='Semantic Loss', linewidth=1.5)
    axes[0].plot(epochs, bd_losses, 'r:', label='Boundary Loss', linewidth=1.5)
    axes[0].set_title('PIDNet-S Training Loss Curves', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    # Metric plot
    if len(val_epochs) > 0:
        axes[1].plot(val_epochs, val_mIoUs, 'm-o', label='Val mIoU (%)', linewidth=2, markersize=5)
        if any(val_accs):
            axes[1].plot(val_epochs, val_accs, 'c--s', label='Val Pixel Acc (%)', linewidth=1.5, markersize=4)
        axes[1].set_title('Validation Accuracy vs. Epoch', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('Accuracy (%)', fontsize=10)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(loc='lower right')
    else:
        axes[1].text(0.5, 0.5, 'Validation metrics not recorded', ha='center', va='center', fontsize=11)
        axes[1].set_title('Validation Metrics', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[INFO] Training loss plot saved to: {save_path}")
    return fig


def run_training(
    data_root: str = "AI512_Project/Data",
    train_list: str = "PIDNet/data/list/nightcity/train.lst",
    val_list: str = "PIDNet/data/list/nightcity/val.lst",
    epochs: int = 30,
    batch_size: int = 4,
    optimizer_name: str = "adamw",
    base_lr: float = 0.001,
    bd_weight: float = 1.0,
    use_class_weights: bool = False,
    crop_h: int = 512,
    crop_w: int = 1024,
    val_freq: int = 2,
    val_samples: Optional[int] = None,
    num_train_samples: Optional[int] = None,
    save_dir: str = "output/nightcity_pidnet_train",
    pretrained_weights: Optional[str] = None
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("PIDNET-S BASELINE TRAINING ON NIGHTCITY DATASET")
    print("=" * 70)
    print(f"[INFO] Compute Device: {device}")
    print(f"[INFO] Train Samples List: {train_list}")
    print(f"[INFO] Val Samples List:   {val_list}")
    print(f"[INFO] Epochs: {epochs} | Batch Size: {batch_size} | Opt: {optimizer_name} | LR: {base_lr} | BD Weight: {bd_weight}")
    print(f"[INFO] Crop Size: ({crop_h}, {crop_w})")

    # 1. Datasets and Loaders
    train_dataset = NightCity(
        root=data_root,
        list_path=train_list,
        num_classes=19,
        multi_scale=False,
        flip=True,
        crop_size=(crop_h, crop_w)
    )
    val_dataset = NightCity(
        root=data_root,
        list_path=val_list,
        num_classes=19,
        multi_scale=False,
        flip=False,
        crop_size=(512, 1024)
    )

    if num_train_samples and num_train_samples < len(train_dataset):
        train_dataset.files = train_dataset.files[:num_train_samples]

    num_workers = 2 if device.type == 'cuda' else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    print(f"[INFO] Loaded Train Samples: {len(train_dataset):,}")
    print(f"[INFO] Loaded Val Samples:   {len(val_dataset):,}")

    # 2. Build PIDNet-S Model
    model = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128, augment=True)
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"[INFO] Loading backbone pretrained weights from {pretrained_weights}")
        ckpt = torch.load(pretrained_weights, map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        model_dict = model.state_dict()
        state = {k.replace('module.', ''): v for k, v in state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(state)
        model.load_state_dict(model_dict, strict=False)
        print(f"[INFO] Transferred {len(state)} layers into PIDNet-S backbone.")

    model.to(device)

    # 3. Loss Functions
    class_weights = train_dataset.class_weights.to(device) if use_class_weights else None
    sem_criterion = OhemCrossEntropy(ignore_label=255, thres=0.9, min_kept=131072, weight=class_weights)
    bd_criterion = BondaryLoss(coeff_bce=20.0)

    wrapper = FullModelWrapper(model, sem_criterion, bd_criterion, bd_weight=bd_weight)

    # 4. Optimizer
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=0.0005,
            nesterov=False
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    max_iters = epochs * len(train_loader)
    best_mIoU = 0.0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        loss_total, loss_sem, loss_bd = train_epoch(
            wrapper, train_loader, optimizer, device, epoch, epochs, base_lr,
            epoch * len(train_loader), max_iters, scaler=scaler
        )
        elapsed = time.time() - t0

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss_total:.4f} (Sem: {loss_sem:.4f}, BD: {loss_bd:.4f}) | Time: {elapsed:.1f}s")

        epoch_record = {
            'epoch': epoch + 1,
            'train_loss_total': float(loss_total),
            'train_loss_sem': float(loss_sem),
            'train_loss_bd': float(loss_bd)
        }

        # Validation
        if (epoch + 1) % val_freq == 0 or (epoch + 1) == epochs:
            model.augment = False
            metrics = validate_model(model, val_loader, device, num_samples=val_samples)
            model.augment = True

            val_mIoU = metrics['mIoU'] * 100
            val_dice = metrics['mean_Dice'] * 100
            val_sens = metrics['mean_Sensitivity'] * 100
            val_spec = metrics['mean_Specificity'] * 100
            val_acc = metrics['Pixel_Accuracy'] * 100

            print(f"  -> Validation mIoU: {val_mIoU:.2f}% | Pixel Acc: {val_acc:.2f}%")

            is_best = val_mIoU > best_mIoU
            if is_best:
                best_mIoU = val_mIoU
                best_path = os.path.join(save_dir, "best_pidnet_nightcity.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mIoU': best_mIoU,
                    'metrics': metrics
                }, best_path)
                print(f"  [CHECKPOINT] New best model saved to: {best_path}")

            epoch_record.update({
                'val_mIoU': val_mIoU,
                'val_dice': val_dice,
                'val_sensitivity': val_sens,
                'val_specificity': val_spec,
                'val_acc': val_acc
            })

        history.append(epoch_record)

    # Save latest checkpoint & history
    latest_path = os.path.join(save_dir, "latest_pidnet_nightcity.pt")
    torch.save({'epoch': epochs, 'state_dict': model.state_dict(), 'history': history}, latest_path)

    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history
