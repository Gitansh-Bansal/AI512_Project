# ------------------------------------------------------------------------------
# Task 3 Point 5: DDRNet-23-slim Training Engine on NightCity Dataset
# Implements composite segmentation loss: Main Head + 0.4 * Aux Head (OHEM/CE)
# Supports CPU and CUDA execution with checkpointing and validation tracking.
# ------------------------------------------------------------------------------

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from datasets.nightcity import NightCity
from metrics import SegmentationMetrics, CITYSCAPES_CLASSES


class OhemCrossEntropy(nn.Module):
    """
    Online Hard Example Mining Cross Entropy Loss for semantic segmentation.
    Keeps at least min_kept hardest pixels and thresholded pixels.
    """
    def __init__(self, ignore_label: int = 255, thres: float = 0.7, min_kept: int = 100000, weight=None):
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
            loss_sorted, _ = torch.sort(loss, descending=True)
            threshold_val = loss_sorted[min(self.min_kept, loss_sorted.numel() - 1)]
            if threshold_val > self.thresh:
                loss = loss[loss >= self.thresh]
            else:
                loss = loss_sorted[:self.min_kept]
        return loss.mean()


class DDRNetLossWrapper(nn.Module):
    """
    Computes composite loss for DDRNet:
    Loss = Loss_main(pred_main, target) + aux_weight * Loss_aux(pred_aux, target)
    """
    def __init__(self, num_classes: int = 19, ignore_label: int = 255, aux_weight: float = 0.4, use_ohem: bool = True, weight=None):
        super(DDRNetLossWrapper, self).__init__()
        self.aux_weight = aux_weight
        if use_ohem:
            self.criterion = OhemCrossEntropy(ignore_label=ignore_label, thres=0.7, min_kept=50000, weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def forward(self, preds: List[torch.Tensor], target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_main, pred_aux = preds[0], preds[1]
        target_h, target_w = target.shape[-2:]

        if pred_main.shape[-2:] != (target_h, target_w):
            pred_main = F.interpolate(pred_main, size=(target_h, target_w), mode='bilinear', align_corners=True)
        if pred_aux.shape[-2:] != (target_h, target_w):
            pred_aux = F.interpolate(pred_aux, size=(target_h, target_w), mode='bilinear', align_corners=True)

        loss_main = self.criterion(pred_main, target)
        loss_aux = self.criterion(pred_aux, target)
        loss_total = loss_main + self.aux_weight * loss_aux
        return loss_total, loss_main, loss_aux


def train_epoch(model, loss_module, loader, optimizer, device, epoch, max_epochs, base_lr, global_step, max_iters):
    model.train()
    total_loss_accum = 0.0
    main_loss_accum = 0.0
    aux_loss_accum = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
    for step, batch in enumerate(pbar):
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.long().to(device)

        # Poly learning rate decay
        cur_iter = global_step + step
        poly_lr = base_lr * ((1.0 - cur_iter / max_iters) ** 0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = poly_lr

        optimizer.zero_grad()
        preds = model(images)
        loss_total, loss_main, loss_aux = loss_module(preds, labels)

        loss_total.backward()
        optimizer.step()

        total_loss_accum += loss_total.item()
        main_loss_accum += loss_main.item()
        aux_loss_accum += loss_aux.item()
        num_batches += 1

        pbar.set_postfix({
            'Total': f"{loss_total.item():.4f}",
            'Main': f"{loss_main.item():.4f}",
            'Aux': f"{loss_aux.item():.4f}",
            'LR': f"{poly_lr:.6f}"
        })

    avg_total = total_loss_accum / max(1, num_batches)
    avg_main = main_loss_accum / max(1, num_batches)
    avg_aux = aux_loss_accum / max(1, num_batches)
    return avg_total, avg_main, avg_aux


def validate_model(model, val_loader, device, num_samples=None):
    model.eval()
    metric_engine = SegmentationMetrics(num_classes=19, ignore_label=255, class_names=CITYSCAPES_CLASSES)
    eval_count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = batch[0], batch[1]
            images = images.to(device)
            preds = model(images)
            if isinstance(preds, list):
                preds = preds[0]

            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=True)

            pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
            targets = labels.numpy()
            metric_engine.update(pred_classes, targets)

            eval_count += images.size(0)
            if num_samples and eval_count >= num_samples:
                break

    metrics = metric_engine.compute_metrics()
    return metrics


def plot_training_history(history: List[Dict], save_path: Optional[str] = None):
    """
    Plots DDRNet training loss vs epoch (total, main, aux) and validation mIoU vs epoch.
    """
    if not history:
        print("[WARNING] Empty history; cannot generate loss plot.")
        return None

    epochs = [h['epoch'] for h in history]
    total_losses = [h['train_loss_total'] for h in history]
    main_losses = [h['train_loss_main'] for h in history]
    aux_losses = [h['train_loss_aux'] for h in history]
    val_epochs = [h['epoch'] for h in history if 'val_mIoU' in h]
    val_mious = [h['val_mIoU'] for h in history if 'val_mIoU' in h]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Loss vs Epoch
    axes[0].plot(epochs, total_losses, marker='o', color='#1f77b4', linewidth=2, label='Total Loss')
    axes[0].plot(epochs, main_losses, marker='s', linestyle='--', color='#2ca02c', linewidth=1.5, label='Main Head Loss')
    axes[0].plot(epochs, aux_losses, marker='^', linestyle=':', color='#ff7f0e', linewidth=1.5, label='Aux Head Loss')
    axes[0].set_title('DDRNet Training Loss vs. Epoch', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss Value', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    if len(epochs) == 1:
        axes[0].set_xlim(0.5, 1.5)
        axes[0].set_xticks([1])
    elif len(epochs) <= 10:
        axes[0].set_xticks(epochs)

    # Subplot 2: Validation Metric vs Epoch
    if val_mious:
        axes[1].plot(val_epochs, val_mious, marker='o', color='#d62728', linewidth=2, label='Validation mIoU (%)')
        axes[1].set_title('Validation mIoU vs. Epoch', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('mIoU (%)', fontsize=10)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(loc='lower right')
        if len(val_epochs) == 1:
            axes[1].set_xlim(0.5, 1.5)
            axes[1].set_xticks([1])
        elif len(val_epochs) <= 10:
            axes[1].set_xticks(val_epochs)
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
    train_list: str = "DDRNet/data/list/nightcity/train.lst",
    val_list: str = "DDRNet/data/list/nightcity/val.lst",
    epochs: int = 10,
    batch_size: int = 4,
    optimizer_name: str = "sgd",
    base_lr: float = 0.005,
    aux_weight: float = 0.8,
    use_class_weights: bool = True,
    crop_h: int = 512,
    crop_w: int = 512,
    val_freq: int = 1,
    val_samples: Optional[int] = 50,
    num_train_samples: Optional[int] = None,
    save_dir: str = "output/nightcity_ddrnet_train",
    pretrained_weights: Optional[str] = None
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("DDRNET-23-SLIM TRAINING ON NIGHTCITY DATASET")
    print("=" * 70)
    print(f"[INFO] Compute Device: {device}")
    print(f"[INFO] Train Samples List: {train_list}")
    print(f"[INFO] Val Samples List:   {val_list}")
    print(f"[INFO] Epochs: {epochs} | Batch Size: {batch_size} | Opt: {optimizer_name} | LR: {base_lr} | Aux: {aux_weight} | Class Weights: {use_class_weights}")
    print(f"[INFO] Crop Size: ({crop_h}, {crop_w})")

    # 1. Datasets and Loaders
    train_dataset = NightCity(
        root=data_root,
        list_path=train_list,
        num_classes=19,
        multi_scale=True,
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
        crop_size=(crop_h, crop_w)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"[INFO] Train images loaded: {len(train_dataset)} | Batches per epoch: {len(train_loader)}")
    print(f"[INFO] Val images loaded:   {len(val_dataset)}")

    # 2. Build DDRNet-23-slim Model
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64, augment=True)
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"[INFO] Loading backbone pretrained weights from {pretrained_weights}")
        ckpt = torch.load(pretrained_weights, map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        model_dict = model.state_dict()
        state = {k.replace('module.', ''): v for k, v in state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(state)
        model.load_state_dict(model_dict, strict=False)
        print(f"[INFO] Transferred {len(state)} layers into DDRNet-23-slim backbone.")

    model.to(device)

    # 3. Loss & Optimizer
    class_weights = train_dataset.class_weights.to(device) if (use_class_weights and hasattr(train_dataset, 'class_weights')) else None
    loss_module = DDRNetLossWrapper(num_classes=19, ignore_label=255, aux_weight=aux_weight, use_ohem=True, weight=class_weights)
    loss_module.to(device)

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

    max_iters = epochs * len(train_loader)
    best_mIoU = 0.0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        loss_total, loss_main, loss_aux = train_epoch(
            model, loss_module, train_loader, optimizer, device, epoch, epochs, base_lr, epoch * len(train_loader), max_iters
        )
        elapsed = time.time() - t0

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss_total:.4f} (Main: {loss_main:.4f}, Aux: {loss_aux:.4f}) | Time: {elapsed:.1f}s")

        epoch_record = {
            'epoch': epoch + 1,
            'train_loss_total': float(loss_total),
            'train_loss_main': float(loss_main),
            'train_loss_aux': float(loss_aux)
        }

        # Validation
        if (epoch + 1) % val_freq == 0 or (epoch + 1) == epochs:
            print("[INFO] Running Validation...")
            model.augment = False
            metrics = validate_model(model, val_loader, device, num_samples=val_samples)
            model.augment = True

            val_mIoU = metrics['mIoU'] * 100
            val_dice = metrics['mean_Dice'] * 100
            val_sens = metrics['mean_Sensitivity'] * 100
            val_spec = metrics['mean_Specificity'] * 100
            val_acc = metrics['Pixel_Accuracy'] * 100

            print(f"[Val] mIoU: {val_mIoU:.2f}% | Dice: {val_dice:.2f}% | Sensitivity: {val_sens:.2f}% | Specificity: {val_spec:.2f}% | Acc: {val_acc:.2f}%")

            is_best = val_mIoU > best_mIoU
            if is_best:
                best_mIoU = val_mIoU
                best_path = os.path.join(save_dir, "best_ddrnet_nightcity.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mIoU': best_mIoU,
                    'metrics': metrics
                }, best_path)
                print(f"[INFO] New best model saved to: {best_path}")

            epoch_record.update({
                'val_mIoU': val_mIoU,
                'val_dice': val_dice,
                'val_sensitivity': val_sens,
                'val_specificity': val_spec,
                'val_acc': val_acc
            })

        history.append(epoch_record)

    # Save latest checkpoint & history
    latest_path = os.path.join(save_dir, "latest_ddrnet_nightcity.pt")
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
    parser = argparse.ArgumentParser(description="Train DDRNet-23-slim on NightCity")
    parser.add_argument('--data_root', type=str, default='AI512_Project/Data')
    parser.add_argument('--train_list', type=str, default='DDRNet/data/list/nightcity/train.lst')
    parser.add_argument('--val_list', type=str, default='DDRNet/data/list/nightcity/val.lst')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--crop_h', type=int, default=512)
    parser.add_argument('--crop_w', type=int, default=512)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--val_samples', type=int, default=25)
    parser.add_argument('--train_samples', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='output/nightcity_ddrnet_train')
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()

    run_training(
        data_root=args.data_root,
        train_list=args.train_list,
        val_list=args.val_list,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
        crop_h=args.crop_h,
        crop_w=args.crop_w,
        val_freq=args.val_freq,
        val_samples=args.val_samples,
        num_train_samples=args.train_samples,
        save_dir=args.save_dir,
        pretrained_weights=args.pretrained
    )
