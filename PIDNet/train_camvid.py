# ------------------------------------------------------------------------------
# PIDNet-S Training Engine on CamVid Dataset (Original Benchmark Reproduction)
# Implements official PIDNet composite loss:
#   - Semantic OHEM CrossEntropy on Detail (0.4) and Main (1.0) heads
#   - Weighted Binary Cross-Entropy Boundary Loss (coeff_bce=20.0)
#   - Boundary-Aware Semantic (BAS) Loss on high-confidence boundary pixels
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
from datasets.camvid import CamVid
from metrics import SegmentationMetrics, CAMVID_CLASSES
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss


class PIDNetCamVidWrapper(nn.Module):
    """Wraps PIDNet-S with full semantic, boundary, and BAS losses."""

    def __init__(self, model: nn.Module, sem_loss: nn.Module, bd_loss: nn.Module, ignore_label: int = 255):
        super().__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss
        self.ignore_label = ignore_label

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, bd_gt: torch.Tensor):
        outputs = self.model(inputs)  # [x_extra_p, x_, x_extra_d]
        h, w = labels.size(1), labels.size(2)

        for i in range(len(outputs)):
            if outputs[i].size(2) != h or outputs[i].size(3) != w:
                outputs[i] = F.interpolate(
                    outputs[i], size=(h, w), mode='bilinear', align_corners=False
                )

        loss_sem = self.sem_loss(outputs[:-1], labels).mean()
        loss_bd = self.bd_loss(outputs[-1], bd_gt).mean()

        # Boundary-Aware Semantic (BAS) Loss on pixels with P(boundary) > 0.8
        filler = torch.ones_like(labels) * self.ignore_label
        bd_prob = torch.sigmoid(outputs[-1][:, 0, :, :])
        bd_label = torch.where(bd_prob > 0.8, labels, filler)
        if (bd_label != self.ignore_label).any():
            loss_sb = self.sem_loss(outputs[-2], bd_label).mean()
        else:
            loss_sb = torch.zeros(1, device=inputs.device, dtype=loss_sem.dtype).squeeze()

        total_loss = loss_sem + loss_bd + loss_sb
        return total_loss, loss_sem, loss_bd, loss_sb, outputs[1]


def adjust_learning_rate(optimizer, base_lr: float, current_iter: int, max_iters: int, power: float = 0.9) -> float:
    """Polynomial learning rate decay matching HRNet/PIDNet protocol."""
    lr = base_lr * ((1.0 - float(current_iter) / max(1, max_iters)) ** power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_epoch(
    wrapper: PIDNetCamVidWrapper,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    base_lr: float,
    max_iters: int,
):
    wrapper.train()
    total_accum = 0.0
    sem_accum = 0.0
    bd_accum = 0.0
    sb_accum = 0.0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False)
    for batch_idx, (images, labels, bd_gts, _, _) in enumerate(pbar):
        current_iter = epoch * num_batches + batch_idx
        lr = adjust_learning_rate(optimizer, base_lr, current_iter, max_iters)

        images = images.to(device)
        labels = labels.long().to(device)
        bd_gts = bd_gts.float().to(device)

        optimizer.zero_grad()
        loss, loss_sem, loss_bd, loss_sb, _ = wrapper(images, labels, bd_gts)
        loss.backward()
        optimizer.step()

        total_accum += loss.item()
        sem_accum += loss_sem.item()
        bd_accum += loss_bd.item()
        sb_accum += loss_sb.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'sem': f"{loss_sem.item():.3f}",
            'bd': f"{loss_bd.item():.3f}",
            'lr': f"{lr:.5f}"
        })

    denom = max(1, num_batches)
    return total_accum / denom, sem_accum / denom, bd_accum / denom, sb_accum / denom


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_samples: Optional[int] = None,
) -> Dict[str, object]:
    model.eval()
    metric_engine = SegmentationMetrics(num_classes=11, ignore_label=255, class_names=CAMVID_CLASSES)
    eval_count = 0

    with torch.no_grad():
        for images, labels, _, _, _ in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            preds = model(images)
            if isinstance(preds, list):
                preds = preds[1]

            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
            metric_engine.update(pred_classes, labels.numpy())

            eval_count += images.size(0)
            if num_samples and eval_count >= num_samples:
                break

    return metric_engine.compute_metrics()


def plot_training_history(history: List[Dict], save_path: Optional[str] = None):
    """Generates dual-panel Loss vs. Epoch and Validation mIoU vs. Epoch curves."""
    if not history:
        return None

    epochs = [h['epoch'] for h in history]
    total_losses = [h['train_loss_total'] for h in history]
    sem_losses = [h['train_loss_sem'] for h in history]
    bd_losses = [h['train_loss_bd'] for h in history]
    val_epochs = [h['epoch'] for h in history if 'val_mIoU' in h]
    val_mious = [h['val_mIoU'] for h in history if 'val_mIoU' in h]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, total_losses, marker='o', color='#1f77b4', linewidth=2, label='Total Loss')
    axes[0].plot(epochs, sem_losses, marker='s', linestyle='--', color='#2ca02c', linewidth=1.5, label='Semantic OHEM')
    axes[0].plot(epochs, bd_losses, marker='^', linestyle=':', color='#ff7f0e', linewidth=1.5, label='Boundary BCE')
    axes[0].set_title('PIDNet-S on CamVid: Training Loss vs. Epoch', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    if val_mious:
        axes[1].plot(val_epochs, val_mious, marker='o', color='#d62728', linewidth=2, label='Validation mIoU (%)')
        axes[1].set_title('PIDNet-S on CamVid: Validation mIoU vs. Epoch', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mIoU (%)')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[INFO] Loss curve saved to: {save_path}")
    return fig


def run_training(
    data_root: str = "PIDNet/data/",
    train_list: str = "list/camvid/train.lst",
    val_list: str = "list/camvid/val.lst",
    epochs: int = 10,
    batch_size: int = 4,
    optimizer_name: str = "sgd",
    base_lr: float = 0.005,
    crop_h: int = 360,
    crop_w: int = 480,
    val_freq: int = 1,
    val_samples: Optional[int] = None,
    num_train_samples: Optional[int] = None,
    save_dir: str = "output/camvid_pidnet_train",
    pretrained_weights: Optional[str] = None,
):
    if not data_root.endswith(("/", "\\")):
        data_root = data_root + os.sep

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("PIDNET-S TRAINING ON CAMVID DATASET (11 CLASSES)")
    print("=" * 70)
    print(f"[INFO] Device: {device} | Optimizer: {optimizer_name.upper()} | Base LR: {base_lr}")
    print(f"[INFO] Train Split: {train_list} | Val Split: {val_list}")
    print(f"[INFO] Epochs: {epochs} | Batch Size: {batch_size} | Train Crop: ({crop_h}, {crop_w})")

    train_dataset = CamVid(
        root=data_root,
        list_path=train_list,
        num_classes=11,
        multi_scale=True,
        flip=True,
        base_size=crop_w,
        crop_size=(crop_h, crop_w),
    )
    val_dataset = CamVid(
        root=data_root,
        list_path=val_list,
        num_classes=11,
        multi_scale=False,
        flip=False,
        base_size=960,
        crop_size=(720, 960),
    )

    if num_train_samples and num_train_samples < len(train_dataset):
        train_dataset.files = train_dataset.files[:num_train_samples]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    print(f"[INFO] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    model = PIDNet(m=2, n=3, num_classes=11, planes=32, ppm_planes=96, head_planes=128, augment=True)
    if pretrained_weights and os.path.exists(pretrained_weights):
        try:
            ckpt = torch.load(pretrained_weights, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(pretrained_weights, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        model_dict = model.state_dict()
        matched = {
            k.replace('module.', '').replace('model.', ''): v
            for k, v in state.items()
            if k.replace('module.', '').replace('model.', '') in model_dict
            and v.shape == model_dict[k.replace('module.', '').replace('model.', '')].shape
        }
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)
        print(f"[INFO] Loaded {len(matched)} layers from {pretrained_weights}")

    model.to(device)

    sem_criterion = OhemCrossEntropy(ignore_label=255, thres=0.9, min_kept=131072, weight=None)
    bd_criterion = BondaryLoss(coeff_bce=20.0)
    wrapper = PIDNetCamVidWrapper(model, sem_criterion, bd_criterion, ignore_label=255)

    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)

    max_iters = epochs * len(train_loader)
    best_mIoU = -1.0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        loss_total, loss_sem, loss_bd, loss_sb = train_epoch(
            wrapper, train_loader, optimizer, device, epoch, epochs, base_lr, max_iters
        )
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch+1}/{epochs}] Total: {loss_total:.4f} "
            f"(Sem: {loss_sem:.4f}, BD: {loss_bd:.4f}, BAS: {loss_sb:.4f}) | {elapsed:.1f}s"
        )

        record = {
            'epoch': epoch + 1,
            'train_loss_total': float(loss_total),
            'train_loss_sem': float(loss_sem),
            'train_loss_bd': float(loss_bd),
            'train_loss_sb': float(loss_sb),
            'epoch_time_s': float(elapsed),
        }

        if (epoch + 1) % val_freq == 0 or (epoch + 1) == epochs:
            model.augment = False
            metrics = validate_model(model, val_loader, device, num_samples=val_samples)
            model.augment = True

            val_mIoU = metrics['mIoU'] * 100.0
            val_dice = metrics['mean_Dice'] * 100.0
            val_sens = metrics['mean_Sensitivity'] * 100.0
            val_spec = metrics['mean_Specificity'] * 100.0
            val_acc = metrics['Pixel_Accuracy'] * 100.0
            print(
                f"  -> [Val] mIoU: {val_mIoU:.2f}% | Dice: {val_dice:.2f}% | "
                f"Sens: {val_sens:.2f}% | Spec: {val_spec:.2f}% | Pixel Acc: {val_acc:.2f}%"
            )

            if val_mIoU > best_mIoU:
                best_mIoU = val_mIoU
                best_path = os.path.join(save_dir, "best_pidnet_camvid.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mIoU': best_mIoU,
                }, best_path)

            record.update({
                'val_mIoU': val_mIoU,
                'val_dice': val_dice,
                'val_sensitivity': val_sens,
                'val_specificity': val_spec,
                'val_acc': val_acc,
            })

        history.append(record)

    latest_path = os.path.join(save_dir, "latest_pidnet_camvid.pt")
    torch.save({'epoch': epochs, 'state_dict': model.state_dict(), 'history': history}, latest_path)

    with open(os.path.join(save_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history, save_path=os.path.join(save_dir, "loss_vs_epoch.png"))
    print(f"[INFO] Completed. Best Val mIoU: {best_mIoU:.2f}%. Outputs saved to {save_dir}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PIDNet-S on CamVid")
    parser.add_argument('--data_root', type=str, default='PIDNet/data/')
    parser.add_argument('--train_list', type=str, default='list/camvid/train.lst')
    parser.add_argument('--val_list', type=str, default='list/camvid/val.lst')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--crop_h', type=int, default=360)
    parser.add_argument('--crop_w', type=int, default=480)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--val_samples', type=int, default=None)
    parser.add_argument('--train_samples', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='output/camvid_pidnet_train')
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()

    run_training(
        data_root=args.data_root,
        train_list=args.train_list,
        val_list=args.val_list,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        base_lr=args.lr,
        crop_h=args.crop_h,
        crop_w=args.crop_w,
        val_freq=args.val_freq,
        val_samples=args.val_samples,
        num_train_samples=args.train_samples,
        save_dir=args.save_dir,
        pretrained_weights=args.pretrained,
    )
