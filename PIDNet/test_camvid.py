# ------------------------------------------------------------------------------
# PIDNet-S Testing & Inference Pipeline on CamVid Dataset (11 Classes, 960x720)
# Computes full Task 5 metrics: mIoU, Dice, Sensitivity, Specificity, Pixel Acc
# Exports per-class metrics CSV, JSON summary, and colorized predicted overlays.
# ------------------------------------------------------------------------------

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.pidnet import get_pred_model
from datasets.camvid import CamVid
from metrics import SegmentationMetrics, CAMVID_CLASSES, CAMVID_PALETTE


def colorize_camvid_mask(mask: np.ndarray) -> np.ndarray:
    """Converts 2D integer prediction mask (0..10) to RGB CamVid palette."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(CAMVID_PALETTE):
        color_mask[mask == cls_id] = rgb
    return color_mask


def run_test(
    data_root: str = "PIDNet/data/",
    test_list: str = "list/camvid/test.lst",
    weights_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    batch_size: int = 2,
    save_masks: bool = True,
    output_dir: str = "output/camvid_pidnet_test",
) -> Dict[str, object]:
    if not data_root.endswith(("/", "\\")):
        data_root = data_root + os.sep

    os.makedirs(output_dir, exist_ok=True)
    if save_masks:
        masks_dir = os.path.join(output_dir, "predicted_masks")
        overlays_dir = os.path.join(output_dir, "predicted_overlays")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(overlays_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("PIDNET-S EVALUATION ON CAMVID TEST SET (960x720)")
    print("=" * 70)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Test Split: {test_list}")
    print(f"[INFO] Checkpoint: {weights_path if weights_path else 'Random Initialization'}")

    model = get_pred_model(name='pidnet_s', num_classes=11)
    if weights_path and os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        clean_state = {k.replace('module.', '').replace('model.', ''): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
        print(f"[INFO] Loaded checkpoint from {weights_path}")

    model.to(device)
    model.eval()

    test_dataset = CamVid(
        root=data_root,
        list_path=test_list,
        num_classes=11,
        multi_scale=False,
        flip=False,
        base_size=960,
        crop_size=(720, 960),
    )

    if num_samples and num_samples < len(test_dataset):
        test_dataset.files = test_dataset.files[:num_samples]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"[INFO] Evaluating {len(test_dataset)} CamVid test images...")

    metric_engine = SegmentationMetrics(num_classes=11, ignore_label=255, class_names=CAMVID_CLASSES)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    saved_count = 0
    with torch.no_grad():
        for images, labels, _, _, names in tqdm(test_loader, desc="Testing CamVid", leave=False):
            images = images.to(device)
            preds = model(images)
            if isinstance(preds, list):
                preds = preds[1]

            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
            metric_engine.update(pred_classes, labels.numpy())

            if save_masks and saved_count < 10:
                for b in range(images.size(0)):
                    if saved_count < 10:
                        color_mask = colorize_camvid_mask(pred_classes[b])
                        Image.fromarray(color_mask).save(os.path.join(masks_dir, f"{names[b]}_pred.png"))

                        img_np = np.clip((images[b].cpu().permute(1, 2, 0).numpy() * std + mean) * 255.0, 0, 255).astype(np.uint8)
                        overlay = (0.55 * img_np + 0.45 * color_mask).astype(np.uint8)
                        Image.fromarray(overlay).save(os.path.join(overlays_dir, f"{names[b]}_overlay.png"))
                        saved_count += 1

    summary = metric_engine.compute_metrics()
    print("\n" + "=" * 70)
    print("CAMVID TEST EVALUATION RESULTS (PIDNet-S):")
    print("=" * 70)
    print(f"  Mean IoU (mIoU):           {summary['mIoU'] * 100:.2f}%")
    print(f"  Mean Dice Score (F1):      {summary['mean_Dice'] * 100:.2f}%")
    print(f"  Mean Sensitivity (Recall): {summary['mean_Sensitivity'] * 100:.2f}%")
    print(f"  Mean Specificity:          {summary['mean_Specificity'] * 100:.2f}%")
    print(f"  Overall Pixel Accuracy:    {summary['Pixel_Accuracy'] * 100:.2f}%")
    print(f"  Valid Pixels Evaluated:    {summary['Total_Pixels_Evaluated']:,}")

    df_per_class = pd.DataFrame(summary['Per_Class'])
    df_per_class['IoU (%)'] = df_per_class['IoU'] * 100
    df_per_class['Dice (%)'] = df_per_class['Dice'] * 100
    df_per_class['Sensitivity (%)'] = df_per_class['Sensitivity'] * 100
    df_per_class['Specificity (%)'] = df_per_class['Specificity'] * 100

    csv_path = os.path.join(output_dir, "test_per_class_metrics.csv")
    df_per_class.to_csv(csv_path, index=False)

    test_json = {
        'model': 'PIDNet-S',
        'dataset': 'CamVid (11 classes, 960x720)',
        'samples_tested': len(test_dataset),
        'weights_used': str(weights_path),
        'mIoU_percent': round(summary['mIoU'] * 100, 2),
        'mean_Dice_percent': round(summary['mean_Dice'] * 100, 2),
        'mean_Sensitivity_percent': round(summary['mean_Sensitivity'] * 100, 2),
        'mean_Specificity_percent': round(summary['mean_Specificity'] * 100, 2),
        'Pixel_Accuracy_percent': round(summary['Pixel_Accuracy'] * 100, 2),
    }
    json_path = os.path.join(output_dir, "test_metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f, indent=2)

    print(f"[INFO] Metrics exported to: {csv_path} & {json_path}")
    return test_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate PIDNet-S on CamVid")
    parser.add_argument('--data_root', type=str, default='PIDNet/data/')
    parser.add_argument('--test_list', type=str, default='list/camvid/test.lst')
    parser.add_argument('--weights', type=str, default='output/camvid_pidnet_train/best_pidnet_camvid.pt')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='output/camvid_pidnet_test')
    args = parser.parse_args()

    run_test(
        data_root=args.data_root,
        test_list=args.test_list,
        weights_path=args.weights,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
