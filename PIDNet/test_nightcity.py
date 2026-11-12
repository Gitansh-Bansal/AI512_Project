# ------------------------------------------------------------------------------
# Task 3: PIDNet-S Testing & Inference Pipeline on NightCity Dataset
# Evaluates Point 5 metrics (mIoU, Dice, Sensitivity, Specificity, Pixel Accuracy)
# Exports per-class metric breakdown, JSON summary, and colorized semantic masks.
# ------------------------------------------------------------------------------

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.pidnet import get_pred_model
from datasets.nightcity import NightCity
from metrics import SegmentationMetrics, CITYSCAPES_CLASSES
from eval_nightcity import colorize_mask

def run_test(
    data_root: str = "AI512_Project/Data",
    test_list: str = "PIDNet/data/list/nightcity/val.lst",
    weights_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    batch_size: int = 2,
    save_masks: bool = True,
    output_dir: str = "output/nightcity_test_results"
):
    os.makedirs(output_dir, exist_ok=True)
    if save_masks:
        masks_dir = os.path.join(output_dir, "predicted_masks")
        overlays_dir = os.path.join(output_dir, "predicted_overlays")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(overlays_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("PIDNET-S TESTING & INFERENCE PIPELINE ON NIGHTCITY")
    print("=" * 70)
    print(f"[INFO] Compute Device: {device}")
    print(f"[INFO] Test Dataset List: {test_list}")
    print(f"[INFO] Weights Checkpoint: {weights_path if weights_path else 'Baseline Initialization'}")
    print(f"[INFO] Output Directory: {output_dir}")

    # 1. Instantiate Model
    model = get_pred_model(name='pidnet_s', num_classes=19)
    if weights_path and os.path.exists(weights_path):
        print(f"[INFO] Loading weights from: {weights_path}")
        try:
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        clean_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
        print("[INFO] Checkpoint loaded successfully.")
    else:
        print("[INFO] No external checkpoint specified; running on current model initialization.")

    model.to(device)
    model.eval()

    # 2. Dataset and Loader
    test_dataset = NightCity(
        root=data_root,
        list_path=test_list,
        num_classes=19,
        multi_scale=False,
        flip=False,
        crop_size=(512, 1024)
    )

    total_samples = len(test_dataset)
    eval_count = min(num_samples, total_samples) if num_samples else total_samples
    if num_samples and num_samples < total_samples:
        test_dataset.files = test_dataset.files[:eval_count]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"[INFO] Evaluating {len(test_dataset):,} samples...")

    # 3. Testing Loop
    metric_engine = SegmentationMetrics(num_classes=19, ignore_label=255, class_names=CITYSCAPES_CLASSES)
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])

    saved_count = 0
    with torch.no_grad():
        for batch_idx, (images, labels, _, _, names) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            preds = model(images)
            if isinstance(preds, list):
                preds = preds[1]

            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
            targets = labels.numpy()

            metric_engine.update(pred_classes, targets)

            # Export masks for the first 10 samples
            if save_masks and saved_count < 10:
                for b in range(images.size(0)):
                    if saved_count < 10:
                        pred_mask = pred_classes[b]
                        color_mask = colorize_mask(pred_mask)
                        mask_img = Image.fromarray(color_mask)
                        mask_path = os.path.join(masks_dir, f"{names[b]}_pred.png")
                        mask_img.save(mask_path)

                        # Overlay
                        img_np = np.clip((images[b].cpu().permute(1, 2, 0).numpy() * std + mean) * 255.0, 0, 255).astype(np.uint8)
                        overlay = (0.55 * img_np + 0.45 * color_mask).astype(np.uint8)
                        overlay_img = Image.fromarray(overlay)
                        overlay_path = os.path.join(overlays_dir, f"{names[b]}_overlay.png")
                        overlay_img.save(overlay_path)
                        saved_count += 1

    # 4. Compute Metrics
    summary = metric_engine.compute_metrics()

    print("\n" + "=" * 70)
    print("POINT 5 TEST EVALUATION RESULTS:")
    print("=" * 70)
    print(f"  • Mean IoU (mIoU):              {summary['mIoU'] * 100:.2f}%")
    print(f"  • Mean Dice Score (F1):         {summary['mean_Dice'] * 100:.2f}%")
    print(f"  • Mean Sensitivity (Recall):    {summary['mean_Sensitivity'] * 100:.2f}%")
    print(f"  • Mean Specificity:             {summary['mean_Specificity'] * 100:.2f}%")
    print(f"  • Overall Pixel Accuracy:       {summary['Pixel_Accuracy'] * 100:.2f}%")
    print(f"  • Total Valid Pixels Tested:    {summary['Total_Pixels_Evaluated']:,}")

    # Per-Class DataFrame
    df_per_class = pd.DataFrame(summary['Per_Class'])
    df_per_class['IoU (%)'] = df_per_class['IoU'] * 100
    df_per_class['Dice (%)'] = df_per_class['Dice'] * 100
    df_per_class['Sensitivity (%)'] = df_per_class['Sensitivity'] * 100
    df_per_class['Specificity (%)'] = df_per_class['Specificity'] * 100

    csv_path = os.path.join(output_dir, "test_per_class_metrics.csv")
    df_per_class.to_csv(csv_path, index=False)
    print(f"[INFO] Per-class metrics table saved to: {csv_path}")

    # JSON export
    test_json = {
        'model': 'PIDNet-S',
        'checkpoint': weights_path,
        'dataset': 'NightCity',
        'samples_tested': eval_count,
        'mIoU': summary['mIoU'],
        'mean_Dice': summary['mean_Dice'],
        'mean_Sensitivity': summary['mean_Sensitivity'],
        'mean_Specificity': summary['mean_Specificity'],
        'pixel_accuracy': summary['Pixel_Accuracy'],
        'per_class': summary['Per_Class']
    }
    json_path = os.path.join(output_dir, "test_metrics_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_json, f, indent=2)
    print(f"[INFO] Test summary JSON saved to: {json_path}")

    print("=" * 70)
    print("[INFO] Testing and inference completed successfully.")
    print("=" * 70)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test PIDNet-S on NightCity")
    parser.add_argument('--data_root', type=str, default='AI512_Project/Data')
    parser.add_argument('--test_list', type=str, default='PIDNet/data/list/nightcity/val.lst')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save_masks', action='store_true', default=True)
    parser.add_argument('--output_dir', type=str, default='output/nightcity_test_results')
    args = parser.parse_args()

    run_test(
        data_root=args.data_root,
        test_list=args.test_list,
        weights_path=args.weights,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        save_masks=args.save_masks,
        output_dir=args.output_dir
    )
