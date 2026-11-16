# ------------------------------------------------------------------------------
# Task 3: TwinLiteNetPlus Training Engine on NightCity Dataset
# Adapts TwinLiteNetPlus for full 19-class semantic segmentation.
# Optimized with AMP FP16, non-blocking transfers, kthvalue OHEM, and pinned memory.
# Supports CPU and CUDA execution with validation tracking and loss plotting.
# ------------------------------------------------------------------------------

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

from model.twinlitenet_seg import get_twinlitenet_plus_seg
from datasets.nightcity import NightCity
from metrics import SegmentationMetrics, CITYSCAPES_CLASSES


class OhemCrossEntropy(nn.Module):
    """Online Hard Example Mining Cross Entropy Loss for semantic segmentation."""
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
            k = loss.numel() - self.min_kept
            threshold_val = torch.kthvalue(loss, k).values
            if threshold_val > self.thresh:
                loss = loss[loss >= self.thresh]
            else:
                loss = loss[loss >= threshold_val]
        return loss.mean()


def train_epoch(model, criterion, loader, optimizer, device, epoch, max_epochs, base_lr, global_step, max_iters, scaler=None):
    model.train()
    total_loss_accum = 0.0
    num_batches = len(loader)
    use_amp = (scaler is not None and device.type == 'cuda')

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{max_epochs} [TwinLite Train]", leave=False)
    for step, batch in enumerate(pbar):
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        # Poly LR schedule
        cur_iter = global_step + step
        poly_lr = base_lr * ((1.0 - float(cur_iter) / max_iters) ** 0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = poly_lr

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(images)
            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=True)
            loss = criterion(preds, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss_accum += loss.detach()

        if (step + 1) % 15 == 0 or (step + 1) == num_batches:
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{poly_lr:.6f}"
            })

    return float((total_loss_accum / max(1, num_batches)).item())


def validate_model(model, val_loader, device, num_samples=None):
    model.eval()
    metric_engine = SegmentationMetrics(num_classes=19, ignore_label=255, class_names=CITYSCAPES_CLASSES)
    eval_count = 0
    use_amp = (device.type == 'cuda')

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating TwinLite", leave=False):
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(images)
                if preds.shape[-2:] != labels.shape[-2:]:
                    preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=True)

            pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
            targets = labels.numpy()
            metric_engine.update(pred_classes, targets)

            eval_count += images.size(0)
            if num_samples is not None and eval_count >= num_samples:
                break

    return metric_engine.compute_metrics()


def plot_training_history(history: List[Dict], save_path: Optional[str] = None):
    epochs = [r['epoch'] for r in history]
    train_losses = [r['train_loss_total'] for r in history]

    val_epochs = [r['epoch'] for r in history if 'val_mIoU' in r]
    val_mious = [r['val_mIoU'] for r in history if 'val_mIoU' in r]
    val_accs = [r.get('val_acc', 0) for r in history if 'val_acc' in r]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].set_title('TwinLiteNetPlus Training Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    if val_mious:
        axes[1].plot(val_epochs, val_mious, marker='o', color='#2ca02c', linewidth=2, label='Val mIoU (%)')
        if any(val_accs):
            axes[1].plot(val_epochs, val_accs, marker='s', color='#ff7f0e', linestyle='--', linewidth=1.5, label='Pixel Acc (%)')
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
        print(f"[INFO] Plot saved to: {save_path}")
    return fig


def run_training(
    data_root: str = "AI512_Project/Data",
    train_list: str = "TwinLiteNetPlus/data/list/nightcity/train.lst",
    val_list: str = "TwinLiteNetPlus/data/list/nightcity/val.lst",
    model_config: str = "small",
    epochs: int = 30,
    batch_size: int = 4,
    optimizer_name: str = "adamw",
    base_lr: float = 0.001,
    use_class_weights: bool = False,
    crop_h: int = 512,
    crop_w: int = 1024,
    val_freq: int = 2,
    val_samples: Optional[int] = None,
    num_train_samples: Optional[int] = None,
    save_dir: str = "output/nightcity_twinlitenetplus_train",
    pretrained_weights: Optional[str] = None
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("TWINLITENETPLUS BASELINE TRAINING ON NIGHTCITY DATASET")
    print("=" * 70)
    print(f"[INFO] Compute Device: {device}")
    print(f"[INFO] Model Config:   {model_config}")
    print(f"[INFO] Train List:     {train_list}")
    print(f"[INFO] Val List:       {val_list}")
    print(f"[INFO] Epochs: {epochs} | Batch Size: {batch_size} | Opt: {optimizer_name} | LR: {base_lr}")
    print(f"[INFO] Crop Size: ({crop_h}, {crop_w})")

    # 1. Dataset & Loaders
    train_dataset = NightCity(
        root=data_root,
        list_path=train_list,
        num_classes=19,
        multi_scale=False,
        flip=True,
        crop_size=(crop_h, crop_w)
    )
    if num_train_samples and num_train_samples < len(train_dataset):
        train_dataset.files = train_dataset.files[:num_train_samples]

    val_dataset = NightCity(
        root=data_root,
        list_path=val_list,
        num_classes=19,
        multi_scale=False,
        flip=False,
        crop_size=(512, 1024)
    )

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
    print(f"[INFO] Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    # 2. Build Model
    model = get_twinlitenet_plus_seg(config=model_config, num_classes=19)
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"[INFO] Loading weights from {pretrained_weights}")
        ckpt = torch.load(pretrained_weights, map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        model_dict = model.state_dict()
        state = {k.replace('module.', ''): v for k, v in state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(state)
        model.load_state_dict(model_dict, strict=False)
        print(f"[INFO] Loaded {len(state)} compatible weights.")

    model.to(device)

    # 3. Loss & Optimizer
    class_weights = train_dataset.class_weights.to(device) if (use_class_weights and hasattr(train_dataset, 'class_weights')) else None
    criterion = OhemCrossEntropy(ignore_label=255, thres=0.7, min_kept=50000, weight=class_weights)
    criterion.to(device)

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    max_iters = epochs * len(train_loader)
    best_mIoU = 0.0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        loss = train_epoch(
            model, criterion, train_loader, optimizer, device, epoch, epochs, base_lr,
            epoch * len(train_loader), max_iters, scaler=scaler
        )
        elapsed = time.time() - t0

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss:.4f} | Time: {elapsed:.1f}s")

        epoch_record = {
            'epoch': epoch + 1,
            'train_loss_total': float(loss)
        }

        # Validation
        if (epoch + 1) % val_freq == 0 or (epoch + 1) == epochs:
            metrics = validate_model(model, val_loader, device, num_samples=val_samples)

            val_mIoU = metrics['mIoU'] * 100
            val_dice = metrics['mean_Dice'] * 100
            val_sens = metrics['mean_Sensitivity'] * 100
            val_spec = metrics['mean_Specificity'] * 100
            val_acc = metrics['Pixel_Accuracy'] * 100

            print(f"  -> Validation mIoU: {val_mIoU:.2f}% | Pixel Acc: {val_acc:.2f}%")

            is_best = val_mIoU > best_mIoU
            if is_best:
                best_mIoU = val_mIoU
                best_path = os.path.join(save_dir, "best_twinlitenetplus_nightcity.pt")
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

    # Save latest & history
    latest_path = os.path.join(save_dir, "latest_twinlitenetplus_nightcity.pt")
    torch.save({'epoch': epochs, 'state_dict': model.state_dict(), 'history': history}, latest_path)

    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    plot_path = os.path.join(save_dir, "loss_vs_epoch.png")
    plot_training_history(history, save_path=plot_path)

    print("=" * 70)
    print(f"[INFO] Training completed. Best mIoU: {best_mIoU:.2f}%. Checkpoints saved in: {save_dir}")
    print("=" * 70)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train TwinLiteNet+ on NightCity")
    parser.add_argument('--data_root', type=str, default='AI512_Project/Data')
    parser.add_argument('--train_list', type=str, default='TwinLiteNetPlus/data/list/nightcity/train.lst')
    parser.add_argument('--val_list', type=str, default='TwinLiteNetPlus/data/list/nightcity/val.lst')
    args = parser.parse_args()
    run_training(data_root=args.data_root, train_list=args.train_list, val_list=args.val_list)
