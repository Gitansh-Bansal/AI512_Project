# ------------------------------------------------------------------------------
# DDRNet-23-slim Training Engine on CamVid Dataset (11 Classes)
# Implements official DualResNet composite loss: Main Head + aux_weight * Aux Head
# Supports CPU and CUDA execution, Poly LR decay, checkpointing, and loss plotting.
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
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR / "segmentation") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "segmentation"))

from DDRNet_23_slim import DualResNet, BasicBlock
from datasets.camvid import CamVid
from metrics import SegmentationMetrics, CAMVID_CLASSES


class OhemCrossEntropy(nn.Module):
    """Online Hard Example Mining Cross Entropy Loss for semantic segmentation."""

    def __init__(self, ignore_label: int = 255, thres: float = 0.7, min_kept: int = 50000, weight=None):
        super().__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thres)
        self.min_kept = int(min_kept)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def forward(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(score, target).view(-1)
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
    Composite bilateral loss for DDRNet:
    Loss = Loss_main(pred_main, target) + aux_weight * Loss_aux(pred_aux, target)
    Uses align_corners=False as required by official DDRNet protocol.
    """

    def __init__(self, ignore_label: int = 255, aux_weight: float = 0.4, use_ohem: bool = True):
        super().__init__()
        self.aux_weight = aux_weight
        if use_ohem:
            self.criterion = OhemCrossEntropy(ignore_label=ignore_label, thres=0.7, min_kept=50000)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, preds: List[torch.Tensor], target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_main, pred_aux = preds[0], preds[1]
        h, w = target.shape[-2:]

        if pred_main.shape[-2:] != (h, w):
            pred_main = F.interpolate(pred_main, size=(h, w), mode='bilinear', align_corners=False)
        if pred_aux.shape[-2:] != (h, w):
            pred_aux = F.interpolate(pred_aux, size=(h, w), mode='bilinear', align_corners=False)

        loss_main = self.criterion(pred_main, target)
        loss_aux = self.criterion(pred_aux, target)
        loss_total = loss_main + self.aux_weight * loss_aux
        return loss_total, loss_main, loss_aux


def resolve_camvid_root(data_root: Optional[str] = None) -> str:
    """Resolves CamVid root directory adaptively across Kaggle, DDRNet/data, or PIDNet/data."""
    candidates = []
    if data_root:
        candidates.append(Path(data_root))

    if Path("/kaggle/input").exists():
        # Search by file pattern first
        for f in Path("/kaggle/input").rglob("*0001TP*"):
            for p_cand in f.parents:
                if (p_cand / "images").exists() or (p_cand / "train").exists() or p_cand.parent == Path("/kaggle/input"):
                    candidates.append(p_cand)
                    break
            if candidates:
                break
        # Search for any mounted directory matching camvid
        for d in Path("/kaggle/input").glob("*camvid*"):
            if d.is_dir():
                candidates.append(d)

    candidates.extend([
        SCRIPT_DIR / "data",
        REPO_ROOT / "PIDNet" / "data",
        REPO_ROOT / "DDRNet" / "data",
    ])

    for cand in candidates:
        if not cand or not cand.exists():
            continue
        # Check if candidate or any common subfolder has .png images
        for sub in [cand, cand / "camvid", cand / "CamVid", cand / "images", cand / "train"]:
            if sub.exists() and any(sub.glob("*.png")):
                root_str = str(cand.resolve())
                return root_str if root_str.endswith(("/", "\\")) else root_str + os.sep

    # If data_root was explicitly passed and exists, trust it
    if data_root and Path(data_root).exists():
        root_str = str(Path(data_root).resolve())
        return root_str if root_str.endswith(("/", "\\")) else root_str + os.sep

    # Safe fallback to SCRIPT_DIR / "data" without throwing an unrecoverable exception
    fallback = SCRIPT_DIR / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    root_str = str(fallback.resolve())
    return root_str if root_str.endswith(("/", "\\")) else root_str + os.sep


def train_epoch(model, loss_module, loader, optimizer, device, epoch, max_epochs, base_lr, global_step, max_iters):
    model.train()
    total_accum = 0.0
    main_accum = 0.0
    aux_accum = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        images, labels = batch[0].to(device), batch[1].long().to(device)

        cur_iter = global_step + step
        poly_lr = base_lr * ((1.0 - cur_iter / max(1, max_iters)) ** 0.9)
        for pg in optimizer.param_groups:
            pg['lr'] = poly_lr

        optimizer.zero_grad()
        preds = model(images)
        loss_total, loss_main, loss_aux = loss_module(preds, labels)
        loss_total.backward()
        optimizer.step()

        total_accum += loss_total.item()
        main_accum += loss_main.item()
        aux_accum += loss_aux.item()

        pbar.set_postfix({
            'Total': f"{loss_total.item():.3f}",
            'Main': f"{loss_main.item():.3f}",
            'Aux': f"{loss_aux.item():.3f}",
            'LR': f"{poly_lr:.5f}"
        })

    denom = max(1, num_batches)
    return total_accum / denom, main_accum / denom, aux_accum / denom


def validate_model(model, val_loader, device, num_samples: Optional[int] = None) -> Dict[str, object]:
    model.eval()
    metric_engine = SegmentationMetrics(num_classes=11, ignore_label=255, class_names=CAMVID_CLASSES)
    eval_count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = batch[0].to(device), batch[1]
            preds = model(images)
            if isinstance(preds, list):
                preds = preds[0]
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
    main_losses = [h['train_loss_main'] for h in history]
    aux_losses = [h['train_loss_aux'] for h in history]
    val_epochs = [h['epoch'] for h in history if 'val_mIoU' in h]
    val_mious = [h['val_mIoU'] for h in history if 'val_mIoU' in h]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, total_losses, marker='o', color='#1f77b4', linewidth=2, label='Total Loss')
    axes[0].plot(epochs, main_losses, marker='s', linestyle='--', color='#2ca02c', linewidth=1.5, label='Main Head Loss')
    axes[0].plot(epochs, aux_losses, marker='^', linestyle=':', color='#ff7f0e', linewidth=1.5, label='Aux Head Loss')
    axes[0].set_title('DDRNet-23-slim on CamVid: Training Loss vs. Epoch', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    if val_mious:
        axes[1].plot(val_epochs, val_mious, marker='o', color='#d62728', linewidth=2, label='Validation mIoU (%)')
        axes[1].set_title('DDRNet-23-slim on CamVid: Validation mIoU vs. Epoch', fontsize=12, fontweight='bold')
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
    data_root: Optional[str] = None,
    train_list: str = "list/camvid/train.lst",
    val_list: str = "list/camvid/val.lst",
    epochs: int = 10,
    batch_size: int = 4,
    optimizer_name: str = "adamw",
    base_lr: float = 0.001,
    aux_weight: float = 0.8,
    crop_h: int = 288,
    crop_w: int = 384,
    val_freq: int = 1,
    val_samples: Optional[int] = None,
    num_train_samples: Optional[int] = None,
    save_dir: str = "output/camvid_ddrnet_train",
    pretrained_weights: Optional[str] = None,
):
    resolved_root = resolve_camvid_root(data_root)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("DDRNET-23-SLIM TRAINING ON CAMVID DATASET (11 CLASSES)")
    print("=" * 70)
    print(f"[INFO] Device: {device} | Optimizer: {optimizer_name.upper()} | Base LR: {base_lr} | Aux Weight: {aux_weight}")
    print(f"[INFO] CamVid Root: {resolved_root}")
    print(f"[INFO] Train Split: {train_list} | Val Split: {val_list}")
    print(f"[INFO] Epochs: {epochs} | Batch Size: {batch_size} | Crop Size: ({crop_h}, {crop_w})")

    train_dataset = CamVid(
        root=resolved_root,
        list_path=train_list,
        num_classes=11,
        multi_scale=True,
        flip=True,
        base_size=crop_w,
        crop_size=(crop_h, crop_w),
    )
    val_dataset = CamVid(
        root=resolved_root,
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

    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=11, planes=32, spp_planes=128, head_planes=64, augment=True)
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

    loss_module = DDRNetLossWrapper(ignore_label=255, aux_weight=aux_weight, use_ohem=True).to(device)

    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)

    max_iters = epochs * len(train_loader)
    best_mIoU = -1.0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        loss_total, loss_main, loss_aux = train_epoch(
            model, loss_module, train_loader, optimizer, device, epoch, epochs, base_lr, epoch * len(train_loader), max_iters
        )
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch+1}/{epochs}] Total: {loss_total:.4f} "
            f"(Main: {loss_main:.4f}, Aux: {loss_aux:.4f}) | {elapsed:.1f}s"
        )

        record = {
            'epoch': epoch + 1,
            'train_loss_total': float(loss_total),
            'train_loss_main': float(loss_main),
            'train_loss_aux': float(loss_aux),
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
                best_path = os.path.join(save_dir, "best_ddrnet_camvid.pt")
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

    latest_path = os.path.join(save_dir, "latest_ddrnet_camvid.pt")
    torch.save({'epoch': epochs, 'state_dict': model.state_dict(), 'history': history}, latest_path)

    with open(os.path.join(save_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history, save_path=os.path.join(save_dir, "loss_vs_epoch.png"))
    print(f"[INFO] Completed. Best Val mIoU: {best_mIoU:.2f}%. Outputs saved to {save_dir}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DDRNet-23-slim on CamVid")
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--train_list', type=str, default='list/camvid/train.lst')
    parser.add_argument('--val_list', type=str, default='list/camvid/val.lst')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--aux_weight', type=float, default=0.8)
    parser.add_argument('--crop_h', type=int, default=288)
    parser.add_argument('--crop_w', type=int, default=384)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--val_samples', type=int, default=50)
    parser.add_argument('--train_samples', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='output/camvid_ddrnet_train')
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
        aux_weight=args.aux_weight,
        crop_h=args.crop_h,
        crop_w=args.crop_w,
        val_freq=args.val_freq,
        val_samples=args.val_samples,
        num_train_samples=args.train_samples,
        save_dir=args.save_dir,
        pretrained_weights=args.pretrained,
    )
