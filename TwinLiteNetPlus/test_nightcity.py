# ------------------------------------------------------------------------------
# Task 3 Point 5: TwinLiteNetPlus Evaluation and Testing Pipeline
# Evaluates mIoU, Dice Score, Sensitivity, Specificity, and Pixel Accuracy
# Generates colorized masks and blended overlays for visual verification.
# ------------------------------------------------------------------------------

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model.twinlitenet_seg import get_twinlitenet_plus_seg
from datasets.nightcity import NightCity
from metrics import SegmentationMetrics, CITYSCAPES_CLASSES

CITYSCAPES_PALETTE = [
    [128, 64, 128],  # 0: road
    [244, 35, 232],  # 1: sidewalk
    [70, 70, 70],    # 2: building
    [102, 102, 156], # 3: wall
    [190, 153, 153], # 4: fence
    [153, 153, 153], # 5: pole
    [250, 170, 30],  # 6: traffic light
    [220, 220, 0],   # 7: traffic sign
    [107, 142, 35],  # 8: vegetation
    [152, 251, 152], # 9: terrain
    [70, 130, 180],  # 10: sky
    [220, 20, 60],   # 11: person
    [255, 0, 0],     # 12: rider
    [0, 0, 142],     # 13: car
    [0, 0, 70],      # 14: truck
    [0, 60, 100],    # 15: bus
    [0, 80, 100],    # 16: train
    [0, 0, 230],     # 17: motorcycle
    [119, 11, 32]    # 18: bicycle
]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        color_mask[mask == class_id] = color
    return color_mask


def run_testing(
    data_root: str = "AI512_Project/Data",
    test_list: str = "TwinLiteNetPlus/data/list/nightcity/val.lst",
    model_config: str = "small",
    weights_path: Optional[str] = None,
    output_dir: str = "output/nightcity_twinlitenetplus_test_results",
    batch_size: int = 2,
    num_samples: Optional[int] = None,
    save_masks: bool = True
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("TWINLITENET+ EVALUATION AND TESTING PIPELINE")
    print("=" * 70)
    print(f"[INFO] Compute Device:    {device}")
    print(f"[INFO] Model Config:      {model_config}")
    print(f"[INFO] Test Dataset List: {test_list}")
    print(f"[INFO] Weights Path:      {weights_path if weights_path else 'Baseline Initialization'}")
    print(f"[INFO] Output Directory:  {output_dir}")

    # 1. Model setup
    model = get_twinlitenet_plus_seg(config=model_config, num_classes=19)
    if weights_path and os.path.exists(weights_path):
        print(f"[INFO] Loading checkpoint from: {weights_path}")
        try:
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        clean_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
        print("[INFO] Checkpoint loaded successfully.")
    else:
        print("[INFO] Evaluating baseline model initialization.")

    model.to(device)
    model.eval()

    # 2. Dataset and loader
    test_dataset = NightCity(
        root=data_root,
        list_path=test_list,
        num_classes=19,
        multi_scale=False,
        flip=False,
        crop_size=(512, 1024)
    )
    if num_samples and num_samples < len(test_dataset):
        test_dataset.files = test_dataset.files[:num_samples]

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"[INFO] Loaded {len(test_dataset)} testing/validation images.")

    metric_engine = SegmentationMetrics(num_classes=19, ignore_label=255, class_names=CITYSCAPES_CLASSES)
    masks_dir = os.path.join(output_dir, "predicted_masks")
    overlays_dir = os.path.join(output_dir, "blended_overlays")
    if save_masks:
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(overlays_dir, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    total_inference_time = 0.0
    total_images_processed = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=True):
            images, labels, _, _, names = batch
            images = images.to(device)

            t0 = time.time()
            preds = model(images)
            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=True)
            batch_infer_time = time.time() - t0
            total_inference_time += batch_infer_time
            total_images_processed += images.size(0)

            pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
            targets = labels.numpy()
            metric_engine.update(pred_classes, targets)

            if save_masks:
                for b in range(images.size(0)):
                    name = names[b]
                    pred_mask = pred_classes[b]
                    colored_pred = colorize_mask(pred_mask)

                    mask_save_path = os.path.join(masks_dir, f"{name}_pred.png")
                    cv2.imwrite(mask_save_path, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))

                    raw_img = images[b].cpu().permute(1, 2, 0).numpy()
                    raw_img = np.clip((raw_img * std + mean) * 255.0, 0, 255).astype(np.uint8)
                    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
                    colored_pred_bgr = cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR)
                    overlay = cv2.addWeighted(raw_img, 0.6, colored_pred_bgr, 0.4, 0)
                    overlay_save_path = os.path.join(overlays_dir, f"{name}_overlay.png")
                    cv2.imwrite(overlay_save_path, overlay)

    metrics = metric_engine.compute_metrics()
    avg_latency_ms = (total_inference_time / max(1, total_images_processed)) * 1000.0
    throughput_fps = total_images_processed / max(0.001, total_inference_time)

    metrics['Inference_Latency_ms'] = avg_latency_ms
    metrics['Throughput_FPS'] = throughput_fps
    metrics['Images_Evaluated'] = total_images_processed

    print("\n" + "=" * 70)
    print("TWINLITENET+ EVALUATION METRICS SUMMARY (NIGHTCITY VAL):")
    print("=" * 70)
    print(f" - Mean IoU (mIoU):              {metrics['mIoU'] * 100:.2f}%")
    print(f" - Mean Dice Score (F1):         {metrics['mean_Dice'] * 100:.2f}%")
    print(f" - Mean Sensitivity (Recall):    {metrics['mean_Sensitivity'] * 100:.2f}%")
    print(f" - Mean Specificity:             {metrics['mean_Specificity'] * 100:.2f}%")
    print(f" - Overall Pixel Accuracy:       {metrics['Pixel_Accuracy'] * 100:.2f}%")
    print(f" - Average Latency:              {avg_latency_ms:.2f} ms")
    print(f" - Average Throughput:           {throughput_fps:.2f} FPS")
    print("=" * 70)

    json_path = os.path.join(output_dir, "twinlitenetplus_nightcity_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Complete evaluation results saved to: {json_path}")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test TwinLiteNet+ on NightCity")
    parser.add_argument('--data_root', type=str, default='AI512_Project/Data')
    parser.add_argument('--test_list', type=str, default='TwinLiteNetPlus/data/list/nightcity/val.lst')
    parser.add_argument('--config', type=str, default='small')
    parser.add_argument('--weights', type=str, default='output/nightcity_twinlitenetplus_train/best_twinlitenetplus_nightcity.pt')
    parser.add_argument('--output_dir', type=str, default='output/nightcity_twinlitenetplus_test_results')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=25)
    args = parser.parse_args()

    run_testing(
        data_root=args.data_root,
        test_list=args.test_list,
        model_config=args.config,
        weights_path=args.weights,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
