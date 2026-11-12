# ------------------------------------------------------------------------------
# DDRNet-23-slim Evaluation and Testing Pipeline on CamVid (11 Classes, 960x720)
# Computes mIoU, Dice Score, Sensitivity, Specificity, and Pixel Accuracy.
# Supports evaluating both CPU-trained checkpoints and the official DDRNet23s CamVid weights.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR / "segmentation") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "segmentation"))

from DDRNet_23_slim import DualResNet, BasicBlock
from datasets.camvid import CamVid
from metrics import SegmentationMetrics, CAMVID_CLASSES, CAMVID_PALETTE
from train_camvid import resolve_camvid_root

# Maps SegNet-style 11-class indices [Sky, Building, Pole, Road, Sidewalk, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist]
# to PIDNet/CamVid alphabetical class indices [Bicyclist, Building, Car, Pole, Fence, Pedestrian, Road, Sidewalk, SignSymbol, Sky, Tree]
SEGNET_TO_PIDNET = np.array([9, 1, 3, 6, 7, 10, 8, 4, 2, 5, 0], dtype=np.int64)


def colorize_camvid_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(CAMVID_PALETTE):
        color_mask[mask == class_id] = color
    return color_mask


def run_testing(
    data_root: Optional[str] = None,
    test_list: str = "list/camvid/test.lst",
    weights_path: Optional[str] = None,
    output_dir: str = "output/camvid_ddrnet_test",
    batch_size: int = 2,
    num_samples: Optional[int] = None,
    save_masks: bool = True,
    official_mode: bool = False,
) -> Dict[str, object]:
    resolved_root = resolve_camvid_root(data_root)
    os.makedirs(output_dir, exist_ok=True)
    if save_masks:
        masks_dir = os.path.join(output_dir, "predicted_masks")
        overlays_dir = os.path.join(output_dir, "predicted_overlays")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(overlays_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("DDRNET-23-SLIM EVALUATION ON CAMVID TEST SET (960x720)")
    print("=" * 70)
    print(f"[INFO] Device: {device} | Official Checkpoint Mode: {official_mode}")
    print(f"[INFO] Checkpoint: {weights_path}")

    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=11, planes=32, spp_planes=128, head_planes=64, augment=False)
    if weights_path and os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        clean_state = {
            k.replace('module.', '').replace('model.', ''): v
            for k, v in state.items()
            if not k.startswith('loss.')
        }
        model.load_state_dict(clean_state, strict=False)
        print(f"[INFO] Loaded checkpoint: {weights_path}")

    model.to(device)
    model.eval()

    test_dataset = CamVid(
        root=resolved_root,
        list_path=test_list,
        num_classes=11,
        multi_scale=False,
        flip=False,
        base_size=960,
        crop_size=(720, 960),
    )
    if num_samples and num_samples < len(test_dataset):
        test_dataset.files = test_dataset.files[:num_samples]

    metric_engine = SegmentationMetrics(num_classes=11, ignore_label=255, class_names=CAMVID_CLASSES)
    mean_01 = np.array([0.485, 0.456, 0.406])
    std_01 = np.array([0.229, 0.224, 0.225])
    mean_255 = np.array([123.675, 116.28, 103.53], dtype=np.float32)

    saved_count = 0
    if official_mode:
        # Official DDRNet23s_camvid.pth uses RGB [0-255] - ImageNet_mean_255 and SegNet 11-class indexing
        with torch.no_grad():
            for item in tqdm(test_dataset.files, desc="Testing Official DDRNet", leave=False):
                raw_rgb = np.array(Image.open(os.path.join(resolved_root, 'camvid', item['img'])).convert('RGB')).astype(np.float32)
                lbl_rgb = np.array(Image.open(os.path.join(resolved_root, 'camvid', item['label'])).convert('RGB'))
                target = test_dataset.color2label(lbl_rgb)

                x = torch.from_numpy((raw_rgb - mean_255).transpose(2, 0, 1)).unsqueeze(0).to(device)
                preds = model(x)
                if isinstance(preds, list):
                    preds = preds[0]
                preds = F.interpolate(preds, size=target.shape, mode='bilinear', align_corners=False)
                pred_raw = torch.argmax(preds, dim=1)[0].cpu().numpy()
                pred_cls = SEGNET_TO_PIDNET[pred_raw]
                metric_engine.update(pred_cls, target)

                if save_masks and saved_count < 10:
                    color_mask = colorize_camvid_mask(pred_cls)
                    Image.fromarray(color_mask).save(os.path.join(masks_dir, f"{item['name']}_pred.png"))
                    overlay = (0.55 * np.clip(raw_rgb, 0, 255).astype(np.uint8) + 0.45 * color_mask).astype(np.uint8)
                    Image.fromarray(overlay).save(os.path.join(overlays_dir, f"{item['name']}_overlay.png"))
                    saved_count += 1
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for images, labels, _, _, names in tqdm(test_loader, desc="Testing DDRNet", leave=False):
                images = images.to(device)
                preds = model(images)
                if isinstance(preds, list):
                    preds = preds[0]
                if preds.shape[-2:] != labels.shape[-2:]:
                    preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=False)

                pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
                metric_engine.update(pred_classes, labels.numpy())

                if save_masks and saved_count < 10:
                    for b in range(images.size(0)):
                        if saved_count < 10:
                            color_mask = colorize_camvid_mask(pred_classes[b])
                            Image.fromarray(color_mask).save(os.path.join(masks_dir, f"{names[b]}_pred.png"))
                            img_np = np.clip((images[b].cpu().permute(1, 2, 0).numpy() * std_01 + mean_01) * 255.0, 0, 255).astype(np.uint8)
                            overlay = (0.55 * img_np + 0.45 * color_mask).astype(np.uint8)
                            Image.fromarray(overlay).save(os.path.join(overlays_dir, f"{names[b]}_overlay.png"))
                            saved_count += 1

    summary = metric_engine.compute_metrics()
    print("\n" + "=" * 70)
    print("CAMVID TEST EVALUATION RESULTS (DDRNet-23-slim):")
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
        'model': 'DDRNet-23-slim',
        'dataset': 'CamVid (11 classes, 960x720)',
        'samples_tested': len(test_dataset),
        'official_mode': official_mode,
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

    return test_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate DDRNet-23-slim on CamVid")
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--test_list', type=str, default='list/camvid/test.lst')
    parser.add_argument('--weights', type=str, default='output/camvid_ddrnet_train/best_ddrnet_camvid.pt')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='output/camvid_ddrnet_test')
    parser.add_argument('--official_mode', action='store_true')
    args = parser.parse_args()

    run_testing(
        data_root=args.data_root,
        test_list=args.test_list,
        weights_path=args.weights,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        official_mode=args.official_mode,
    )
