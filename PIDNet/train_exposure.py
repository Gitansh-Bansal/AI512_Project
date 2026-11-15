# ------------------------------------------------------------------------------
# Task 6: PIDNet-NightOpt Training Engine (Optimized Exposure & Boundary Handling)
# Combines:
#   1. PIDNetNightOpt (Haar Wavelet DWT-RF Edge Prior + Illumination-Aware Channel Calibration IACC)
#   2. Focal-OHEM Cross-Entropy with Label Smoothing (Stable gradients in shadows)
#   3. CLAHE Contrast-Restoring Augmentations (Enhances faint curb & obstacle edges)
# ------------------------------------------------------------------------------

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.pidnet_nightopt import PIDNetNightOpt, get_nightopt_model
from datasets.nightcity import NightCity
from metrics import SegmentationMetrics, CITYSCAPES_CLASSES
from utils.criterion import BondaryLoss
from utils.focal_ohem_loss import FocalOhemCrossEntropy
from utils.night_clahe_aug import NightContrastAugmentationPipeline


class FullNightOptModelWrapper(nn.Module):
    """
    Wraps PIDNetNightOpt with:
      - Focal-OHEM Semantic Loss (Detail + Main heads) with label smoothing
      - Boundary BCE Loss (Derivative head enhanced with Haar Wavelets)
    """
    def __init__(
        self,
        model: nn.Module,
        sem_loss: nn.Module,
        bd_loss: nn.Module,
        bd_weight: float = 1.0
    ):
        super(FullNightOptModelWrapper, self).__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss
        self.bd_weight = bd_weight

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, bd_gt: torch.Tensor):
        outputs = self.model(inputs)  # [x_extra_p, x_main, x_extra_d]
        h, w = labels.size(1), labels.size(2)

        for i in range(len(outputs)):
            if outputs[i].size(2) != h or outputs[i].size(3) != w:
                outputs[i] = F.interpolate(
                    outputs[i], size=(h, w), mode='bilinear', align_corners=False
                )

        loss_sem = self.sem_loss(outputs[:2], labels)
        loss_bd = self.bd_loss(outputs[2], bd_gt).mean()

        total_loss = loss_sem + self.bd_weight * loss_bd
        return total_loss, loss_sem, loss_bd, outputs[1]


def adjust_learning_rate(optimizer, base_lr, current_iter, max_iters, power=0.9):
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

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train NightOpt]", leave=False)
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

        total_loss_accum += loss.item()
        sem_loss_accum += loss_sem.item()
        bd_loss_accum += loss_bd.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'sem': f"{loss_sem.item():.4f}",
            'bd': f"{loss_bd.item():.4f}",
            'lr': f"{lr:.6f}"
        })

    avg_total = total_loss_accum / max(1, num_batches)
    avg_sem = sem_loss_accum / max(1, num_batches)
    avg_bd = bd_loss_accum / max(1, num_batches)
    return avg_total, avg_sem, avg_bd


def validate_model(model, val_loader, device, num_samples=None):
    model.eval()
    metric_engine = SegmentationMetrics(num_classes=19, ignore_label=255, class_names=CITYSCAPES_CLASSES)
    eval_count = 0
    use_amp = (device.type == 'cuda')

    with torch.no_grad():
        for images, labels, _, _, _ in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(images)
                if isinstance(preds, list):
                    preds = preds[1]

                if preds.shape[-2:] != labels.shape[-2:]:
                    preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
            targets = labels.numpy()
            metric_engine.update(pred_classes, targets)

            eval_count += images.size(0)
            if num_samples and eval_count >= num_samples:
                break

    metrics = metric_engine.compute_metrics()
    return metrics


def plot_exposure_training_history(history: List[Dict], save_path: Optional[str] = None):
    if not history:
        return None

    epochs = [h['epoch'] for h in history]
    total_losses = [h['train_loss_total'] for h in history]
    sem_losses = [h['train_loss_sem'] for h in history]
    bd_losses = [h['train_loss_bd'] for h in history]
    val_epochs = [h['epoch'] for h in history if 'val_mIoU' in h and h['val_mIoU'] is not None]
    val_mious = [h['val_mIoU'] * 100.0 for h in history if 'val_mIoU' in h and h['val_mIoU'] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, total_losses, marker='o', color='#1f77b4', linewidth=2, label='Total Loss')
    axes[0].plot(epochs, sem_losses, marker='s', linestyle='--', color='#2ca02c', linewidth=1.5, label='Focal-OHEM Sem Loss')
    axes[0].plot(epochs, bd_losses, marker='^', linestyle=':', color='#ff7f0e', linewidth=1.5, label='Wavelet-Boundary Loss')
    axes[0].set_title('Training Losses vs. Epoch', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss Value', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    if val_mious:
        axes[1].plot(val_epochs, val_mious, marker='o', color='#d62728', linewidth=2, label='Validation mIoU (%)')
        axes[1].set_title('Validation mIoU vs. Epoch', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('mIoU (%)', fontsize=10)
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


def run_training_exposure(
    data_root: str = "AI512_Project/Data",
    train_list: str = "PIDNet/data/list/nightcity/train.lst",
    val_list: str = "PIDNet/data/list/nightcity/val.lst",
    epochs: int = 30,
    batch_size: int = 4,
    optimizer_name: str = "adamw",
    base_lr: float = 0.001,
    bd_weight: float = 1.0,
    gamma: float = 1.5,
    label_smoothing: float = 0.05,
    use_clahe_aug: bool = True,
    crop_h: int = 512,
    crop_w: int = 512,
    val_freq: int = 2,
    val_samples: Optional[int] = 10,
    num_train_samples: Optional[int] = None,
    save_dir: str = "output/nightcity_pidnet_exposure"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("PIDNET-NIGHTOPT TRAINING (HAAR WAVELET DWT-RF + IACC + FOCAL-OHEM)")
    print("=" * 70)
    print(f"[INFO] Compute Device: {device}")
    print(f"[INFO] Epochs: {epochs} | Batch Size: {batch_size} | LR: {base_lr} | BD Weight: {bd_weight}")
    print(f"[INFO] Focal gamma={gamma}, Label Smoothing={label_smoothing} | CLAHE Augmentations: {use_clahe_aug}")

    photo_aug = NightContrastAugmentationPipeline(p=0.7) if use_clahe_aug else None

    # Datasets
    train_dataset = NightCity(
        root=data_root,
        list_path=train_list,
        num_classes=19,
        multi_scale=True,
        flip=True,
        crop_size=(crop_h, crop_w),
        photo_aug=photo_aug
    )
    val_dataset = NightCity(
        root=data_root,
        list_path=val_list,
        num_classes=19,
        multi_scale=False,
        flip=False,
        crop_size=(512, 1024),
        photo_aug=None
    )

    if num_train_samples and num_train_samples < len(train_dataset):
        train_dataset.files = train_dataset.files[:num_train_samples]

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)

    num_workers = min(4, os.cpu_count() or 2) if device.type == 'cuda' else 0
    pin_memory = (device.type == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4 if device.type == 'cuda' else 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"[INFO] Train Samples: {len(train_dataset):,} | Val Samples: {len(val_dataset):,} | Workers: {num_workers} | PinMem: {pin_memory}")

    # Model: PIDNetNightOpt (Wavelet + IACC)
    model = PIDNetNightOpt(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128, augment=True)
    model.to(device)

    # Criterions
    sem_criterion = FocalOhemCrossEntropy(
        ignore_label=255, gamma=gamma, label_smoothing=label_smoothing,
        thres=0.85, min_kept=131072
    )
    bd_criterion = BondaryLoss(coeff_bce=20.0)

    wrapper = FullNightOptModelWrapper(
        model, sem_criterion, bd_criterion, bd_weight=bd_weight
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    max_iters = epochs * len(train_loader)

    history = []
    best_miou = 0.0
    best_path = os.path.join(save_dir, "best_pidnet_exposure.pt")
    latest_path = os.path.join(save_dir, "latest_pidnet_exposure.pt")

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
            'train_loss_bd': float(loss_bd),
            'time_sec': float(elapsed),
        }

        if (epoch + 1) % val_freq == 0 or (epoch + 1) == epochs:
            model.augment = False
            metrics = validate_model(model, val_loader, device, num_samples=val_samples)
            model.augment = True

            val_miou = metrics['mIoU']
            val_acc = metrics['Pixel_Accuracy']
            epoch_record['val_mIoU'] = float(val_miou)
            epoch_record['val_Pixel_Accuracy'] = float(val_acc)
            print(f"  -> Validation mIoU: {val_miou*100.0:.2f}% | Pixel Acc: {val_acc*100.0:.2f}%")

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_miou': best_miou,
                    'metrics': metrics
                }, best_path)
                print(f"  [CHECKPOINT] New best mIoU ({best_miou*100.0:.2f}%) saved to {best_path}")

        history.append(epoch_record)
        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    torch.save({'epoch': epochs, 'state_dict': model.state_dict(), 'history': history}, latest_path)
    plot_exposure_training_history(history, save_path=os.path.join(save_dir, "loss_vs_epoch.png"))
    return history
