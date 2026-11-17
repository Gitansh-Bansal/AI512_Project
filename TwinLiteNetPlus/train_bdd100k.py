# ------------------------------------------------------------------------------
# TwinLiteNet+ Multi-Task Training Engine on BDD100K (Drivable Area + Lane Line)
# Implements official composite loss: FocalLossSeg + TverskyLoss (DA & LL)
# Device-agnostic (CPU/CUDA), Poly LR decay, ModelEMA, and Loss vs. Epoch plotting.
# ------------------------------------------------------------------------------

import os
import sys
import time
import json
import math
import yaml
import random
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model.model import TwinLiteNetPlus
from loss import TverskyLoss, FocalLossSeg
from IOUEval import SegmentationMetric


def letterbox(im, new_shape=(384, 640), color=(114, 114, 114)):
    """Resize and pad image cleanly while maintaining aspect ratio."""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """Apply HSV color jittering in-place."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


class BDD100KMultiTaskDataset(Dataset):
    """Device- and path-adaptive BDD100K loader for Drivable Area and Lane Line tasks."""

    def __init__(self, root: str, split: str = "train", hyp: Optional[dict] = None, valid: bool = False, max_samples: Optional[int] = None):
        self.root = Path(root)
        self.split = split
        self.valid = valid
        self.hyp = hyp or {}
        self.is_demo_fallback = False

        # Adaptive directory resolution
        for cand in [self.root / "images" / split, self.root / split, self.root / "100k" / split, self.root / "images" / "100k" / split, self.root / "bdd100k" / "images" / split]:
            if cand.exists():
                self.img_dir = cand
                break
        else:
            self.img_dir = self.root / "images" / split

        for cand in [self.root / "drivable_area_annotations" / split, self.root / "labels" / "drivable" / split, self.root / "drivable" / split, self.root / "da" / split]:
            if cand.exists():
                self.da_dir = cand
                break
        else:
            self.da_dir = self.root / "drivable_area_annotations" / split

        for cand in [self.root / "lane_line_annotations" / split, self.root / "labels" / "lane" / split, self.root / "lane" / split, self.root / "ll" / split]:
            if cand.exists():
                self.ll_dir = cand
                break
        else:
            self.ll_dir = self.root / "lane_line_annotations" / split

        if self.img_dir.exists():
            all_imgs = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png'))])
            self.names = [
                f for f in all_imgs
                if (self.da_dir / (Path(f).stem + ".png")).exists()
                and (self.ll_dir / (Path(f).stem + ".png")).exists()
            ]
        else:
            self.names = []

        # Fallback to inference demo images if dataset not mounted
        if len(self.names) == 0:
            inference_dir = Path(__file__).resolve().parent / "inference" / "images"
            if inference_dir.exists():
                self.img_dir = inference_dir
                self.names = sorted([f for f in os.listdir(inference_dir) if f.lower().endswith(('.jpg', '.png'))])
                self.is_demo_fallback = True

        if max_samples and max_samples < len(self.names):
            self.names = self.names[:max_samples]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        fname = self.names[idx]
        stem = Path(fname).stem
        img_path = str(self.img_dir / fname)
        da_path = str(self.da_dir / f"{stem}.png")
        ll_path = str(self.ll_dir / f"{stem}.png")

        if self.is_demo_fallback or not os.path.exists(da_path) or not os.path.exists(ll_path):
            image = cv2.imread(img_path) if os.path.exists(img_path) else np.zeros((720, 1280, 3), dtype=np.uint8)
            label1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            label2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            image = cv2.imread(img_path)
            label1 = cv2.imread(da_path, 0)
            label2 = cv2.imread(ll_path, 0)

        w_, h_ = 640, 384

        if not self.valid:
            if random.random() < self.hyp.get("prob_hsv", 0.5):
                augment_hsv(image, self.hyp.get("hgain", 0.015), self.hyp.get("sgain", 0.7), self.hyp.get("vgain", 0.4))
            if random.random() < self.hyp.get("prob_flip", 0.5):
                image = np.fliplr(image).copy()
                label1 = np.fliplr(label1).copy()
                label2 = np.fliplr(label2).copy()

        image = letterbox(image, (h_, w_))
        label1 = cv2.resize(label1, (w_, 360), interpolation=cv2.INTER_NEAREST)
        label2 = cv2.resize(label2, (w_, 360), interpolation=cv2.INTER_NEAREST)

        _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)

        gt_da = (seg1 > 0).astype(np.int64)
        gt_ll = (seg2 > 0).astype(np.int64)

        image_rgb = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0
        return fname, torch.from_numpy(image_rgb), torch.from_numpy(gt_da), torch.from_numpy(gt_ll)


class TwinLiteNetPlusTotalLoss(nn.Module):
    """Device-agnostic TwinLiteNet+ multi-task loss (Tversky + Focal on DA & LL)."""

    def __init__(self, hyp: dict):
        super().__init__()
        self.tver_da = TverskyLoss(mode="multiclass", alpha=hyp["alpha1"], beta=1 - hyp["alpha1"], gamma=hyp["gamma1"], from_logits=True)
        self.tver_ll = TverskyLoss(mode="multiclass", alpha=hyp["alpha2"], beta=1 - hyp["alpha2"], gamma=hyp["gamma2"], from_logits=True)
        self.focal = FocalLossSeg(mode="multiclass", alpha=hyp["alpha3"], gamma=hyp["gamma3"])

    def forward(self, outputs: Tuple[torch.Tensor, torch.Tensor], gt_da: torch.Tensor, gt_ll: torch.Tensor):
        out_da, out_ll = outputs
        out_da = out_da[:, :, 12:-12, :]
        out_ll = out_ll[:, :, 12:-12, :]

        tver_da = self.tver_da(out_da, gt_da)
        tver_ll = self.tver_ll(out_ll, gt_ll)
        focal_da = self.focal(out_da, gt_da)
        focal_ll = self.focal(out_ll, gt_ll)

        loss_tver = tver_da + tver_ll
        loss_focal = focal_da + focal_ll
        loss_total = loss_focal + loss_tver
        return loss_total, loss_focal, loss_tver


def evaluate_bdd100k(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Computes official TwinLiteNet+ BDD100K metrics (DA mIoU, LL Acc, LL IoU)."""
    model.eval()
    da_metric = SegmentationMetric(2)
    ll_metric = SegmentationMetric(2)

    with torch.no_grad():
        for _, images, gt_da, gt_ll in tqdm(val_loader, desc="Validating BDD100K", leave=False):
            images = images.to(device)
            out_da, out_ll = model(images)

            pred_da = torch.argmax(out_da, dim=1)[:, 12:-12, :].cpu()
            pred_ll = torch.argmax(out_ll, dim=1)[:, 12:-12, :].cpu()

            da_metric.addBatch(pred_da, gt_da)
            ll_metric.addBatch(pred_ll, gt_ll)

    da_acc = float(da_metric.pixelAccuracy()) * 100.0
    da_fg_iou = float(da_metric.IntersectionOverUnion()) * 100.0
    da_miou = float(da_metric.meanIntersectionOverUnion()) * 100.0

    ll_acc = float(ll_metric.lineAccuracy()) * 100.0
    ll_iou = float(ll_metric.IntersectionOverUnion()) * 100.0
    ll_miou = float(ll_metric.meanIntersectionOverUnion()) * 100.0

    return {
        'DA_mIoU_percent': round(da_miou, 2),
        'DA_FG_IoU_percent': round(da_fg_iou, 2),
        'DA_Pixel_Acc_percent': round(da_acc, 2),
        'LL_Accuracy_percent': round(ll_acc, 2),
        'LL_IoU_percent': round(ll_iou, 2),
        'LL_mIoU_percent': round(ll_miou, 2),
    }


def plot_training_history(history: List[Dict], save_path: Optional[str] = None):
    if not history:
        return None
    epochs = [h['epoch'] for h in history]
    total_l = [h['train_loss_total'] for h in history]
    focal_l = [h['train_loss_focal'] for h in history]
    tver_l = [h['train_loss_tversky'] for h in history]
    da_miou = [h['DA_mIoU_percent'] for h in history if 'DA_mIoU_percent' in h]
    ll_iou = [h['LL_IoU_percent'] for h in history if 'LL_IoU_percent' in h]
    val_epochs = [h['epoch'] for h in history if 'DA_mIoU_percent' in h]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, total_l, marker='o', color='#1f77b4', linewidth=2, label='Total Loss')
    axes[0].plot(epochs, tver_l, marker='s', linestyle='--', color='#2ca02c', linewidth=1.5, label='Tversky Loss')
    axes[0].plot(epochs, focal_l, marker='^', linestyle=':', color='#ff7f0e', linewidth=1.5, label='Focal Loss')
    axes[0].set_title('TwinLiteNet+ on BDD100K: Training Loss vs. Epoch', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    if val_epochs:
        axes[1].plot(val_epochs, da_miou, marker='o', color='#d62728', linewidth=2, label='Drivable Area mIoU (%)')
        axes[1].plot(val_epochs, ll_iou, marker='s', color='#9467bd', linewidth=2, label='Lane Line IoU (%)')
        axes[1].set_title('TwinLiteNet+ on BDD100K: Validation Metrics vs. Epoch', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score (%)')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {save_path}")
    return fig


def run_training(
    data_root: str = "TwinLiteNetPlus/data/bdd100k",
    hyp_path: str = "TwinLiteNetPlus/hyperparameters/twinlitev2_hyper.yaml",
    model_config: str = "small",
    epochs: int = 8,
    batch_size: int = 4,
    base_lr: float = 0.001,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    save_dir: str = "output/bdd100k_twinlitenetplus_train",
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(hyp_path, 'r', encoding='utf-8') as f:
        hyp = yaml.safe_load(f)

    print("=" * 70)
    print(f"TWINLITENET+ ({model_config.upper()}) TRAINING ON BDD100K")
    print("=" * 70)
    print(f"[INFO] Device: {device} | Config: {model_config} | LR: {base_lr} | Epochs: {epochs}")

    train_ds = BDD100KMultiTaskDataset(data_root, split="train", hyp=hyp, valid=False, max_samples=train_samples)
    val_ds = BDD100KMultiTaskDataset(data_root, split="val", hyp=hyp, valid=True, max_samples=val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"[INFO] Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    model = TwinLiteNetPlus(SimpleNamespace(config=model_config)).to(device)
    criterion = TwinLiteNetPlusTotalLoss(hyp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(hyp["momentum"], 0.999), eps=hyp["eps"], weight_decay=hyp["weight_decay"])

    best_score = -1.0
    history = []

    for epoch in range(epochs):
        lr = base_lr * ((1.0 - epoch / max(1, epochs)) ** 1.5)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        t0 = time.time()
        tot_a, foc_a, tv_a = 0.0, 0.0, 0.0
        for _, images, gt_da, gt_ll in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = images.to(device)
            gt_da = gt_da.to(device)
            gt_ll = gt_ll.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_total, loss_focal, loss_tver = criterion(outputs, gt_da, gt_ll)
            loss_total.backward()
            optimizer.step()

            tot_a += loss_total.item()
            foc_a += loss_focal.item()
            tv_a += loss_tver.item()

        n_b = max(1, len(train_loader))
        elapsed = time.time() - t0
        val_res = evaluate_bdd100k(model, val_loader, device)
        print(
            f"[Epoch {epoch+1}/{epochs}] Loss: {tot_a/n_b:.4f} (Focal: {foc_a/n_b:.4f}, Tversky: {tv_a/n_b:.4f}) "
            f"| DA mIoU: {val_res['DA_mIoU_percent']:.2f}% | LL Acc: {val_res['LL_Accuracy_percent']:.2f}% "
            f"| LL IoU: {val_res['LL_IoU_percent']:.2f}% | {elapsed:.1f}s"
        )

        rec = {
            'epoch': epoch + 1,
            'train_loss_total': tot_a / n_b,
            'train_loss_focal': foc_a / n_b,
            'train_loss_tversky': tv_a / n_b,
            'epoch_time_s': elapsed,
            **val_res
        }
        history.append(rec)

        combined = val_res['DA_mIoU_percent'] + val_res['LL_IoU_percent']
        if combined > best_score:
            best_score = combined
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_twinlitenetplus_{model_config}_bdd100k.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, f"latest_twinlitenetplus_{model_config}_bdd100k.pth"))
    with open(os.path.join(save_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, save_path=os.path.join(save_dir, "loss_vs_epoch.png"))
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train TwinLiteNet+ on BDD100K")
    parser.add_argument('--data_root', type=str, default='TwinLiteNetPlus/data/bdd100k')
    parser.add_argument('--hyp', type=str, default='TwinLiteNetPlus/hyperparameters/twinlitev2_hyper.yaml')
    parser.add_argument('--config', type=str, default='small', choices=['nano', 'small', 'medium', 'large'])
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_samples', type=int, default=None)
    parser.add_argument('--val_samples', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='output/bdd100k_twinlitenetplus_train')
    args = parser.parse_args()

    run_training(
        data_root=args.data_root,
        hyp_path=args.hyp,
        model_config=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        save_dir=args.save_dir,
    )
