# ------------------------------------------------------------------------------
# Task 3 Point 5: Comprehensive Baseline Evaluation & Profiling on NightCity
# Evaluates: mIoU, Dice Score, Sensitivity, Specificity, Params, FLOPs, Latency & FPS
# ------------------------------------------------------------------------------

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
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
from torch.utils.data import DataLoader

# Add PIDNet root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.pidnet import get_pred_model
from datasets.nightcity import NightCity
from metrics import SegmentationMetrics, CITYSCAPES_CLASSES
from benchmarks import get_hardware_info, profile_parameters, profile_flops, benchmark_inference_speed

# Official Cityscapes 19-class Color Palette for Visualization
CITYSCAPES_PALETTE = np.array([
    [128,  64, 128],  # 0: road
    [244,  35, 232],  # 1: sidewalk
    [ 70,  70,  70],  # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170,  30],  # 6: traffic light
    [220, 220,   0],  # 7: traffic sign
    [107, 142,  35],  # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [ 70, 130, 180],  # 10: sky
    [220,  20,  60],  # 11: person
    [255,   0,   0],  # 12: rider
    [  0,   0, 142],  # 13: car
    [  0,   0,  70],  # 14: truck
    [  0,  60, 100],  # 15: bus
    [  0,  80, 100],  # 16: train
    [  0,   0, 230],  # 17: motorcycle
    [119,  11,  32],  # 18: bicycle
], dtype=np.uint8)

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Colorizes a 2D integer segmentation mask into RGB."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(CITYSCAPES_PALETTE)):
        color_mask[mask == c] = CITYSCAPES_PALETTE[c]
    return color_mask

def run_evaluation(
    data_root: str = "AI512_Project/Data",
    list_path: str = "PIDNet/data/list/nightcity/val.lst",
    weights_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    batch_size: int = 2,
    output_dir: str = "output/task3_baseline"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("TASK 3: PIDNET-S BASELINE EVALUATION ON NIGHTCITY DATASET")
    print("=" * 70)
    
    # 1. Hardware Specs & Profiling
    hw_info = get_hardware_info(device)
    print("\nHARDWARE PLATFORM PROFILE:")
    for k, v in hw_info.items():
        print(f"  - {k:20s}: {v}")
        
    # 2. Instantiate PIDNet-S
    print("\nINSTANTIATING PIDNET-S MODEL...")
    model = get_pred_model(name='pidnet_s', num_classes=19)
    
    if weights_path and os.path.exists(weights_path):
        print(f"  - Loading checkpoint from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        # strip module. or prefix if needed
        clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)
        print("  [INFO] Checkpoint loaded successfully!")
    else:
        print("  [WARNING] No external checkpoint provided; evaluating baseline initialization.")
        
    model.to(device)
    model.eval()

    # 3. Model Complexity (Parameters & FLOPs)
    print("\nPROFILING MODEL COMPLEXITY (Parameters & FLOPs):")
    param_info = profile_parameters(model)
    print(f"  - Total Parameters:     {param_info['total_params_M']:.2f} M ({param_info['total_params_raw']:,} params)")
    print(f"  - Trainable Parameters: {param_info['trainable_params_M']:.2f} M")
    
    flop_resolutions = [(512, 512), (512, 1024), (1024, 2048)]
    flop_results = profile_flops(model, flop_resolutions, device=device)
    for res_info in flop_results:
        print(f"  - Resolution {res_info['Resolution']:9s} | MACs: {res_info['MACs_G']:.2f} G | Estimated FLOPs: {res_info['FLOPs_G']:.2f} G")

    # 4. Inference Latency & FPS Benchmarking
    print("\nBENCHMARKING INFERENCE SPEED & LATENCY:")
    # Run speed benchmark at (512, 1024)
    speed_results = benchmark_inference_speed(
        model,
        input_resolution=(512, 1024),
        batch_size=1,
        device=device,
        warmup_runs=10,
        timed_runs=25
    )
    print(f"  - Evaluation Resolution:  {speed_results['Resolution']}")
    print(f"  - Mean Latency:           {speed_results['Mean_Latency_ms']:.2f} ms ± {speed_results['Std_Latency_ms']:.2f} ms")
    print(f"  - Median Latency:         {speed_results['Median_Latency_ms']:.2f} ms (Min: {speed_results['Min_Latency_ms']:.2f} ms, Max: {speed_results['Max_Latency_ms']:.2f} ms)")
    print(f"  - Throughput:             {speed_results['FPS']:.2f} FPS")

    # 5. NightCity Validation Dataset & DataLoader
    print("\nINITIALIZING NIGHTCITY VALIDATION DATASET...")
    val_dataset = NightCity(
        root=data_root,
        list_path=list_path,
        num_classes=19,
        multi_scale=False,
        flip=False,
        crop_size=(512, 1024)
    )
    
    total_val_samples = len(val_dataset)
    eval_count = min(num_samples, total_val_samples) if num_samples else total_val_samples
    print(f"  - Total Validation Images: {total_val_samples:,}")
    print(f"  - Samples to Evaluate:     {eval_count:,}")
    
    # Optional subset sampler
    if num_samples and num_samples < total_val_samples:
        indices = list(range(eval_count))
        val_dataset.files = [val_dataset.files[i] for i in indices]

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 6. Running Point 5 Semantic Segmentation Metrics
    print("\nRUNNING ACCURACY & SEGMENTATION METRICS EVALUATION...")
    metric_engine = SegmentationMetrics(num_classes=19, ignore_label=255, class_names=CITYSCAPES_CLASSES)
    
    saved_visualizations = 0
    vis_samples = []

    with torch.no_grad():
        for batch_idx, (images, labels, _, _, names) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = images.to(device)
            # Forward pass: PIDNet-S test inference
            preds = model(images)  # (B, 19, H, W)
            
            # Upsample if needed
            if preds.shape[-2:] != labels.shape[-2:]:
                preds = nn.functional.interpolate(
                    preds, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )
            
            pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
            targets = labels.numpy()

            metric_engine.update(pred_classes, targets)

            # Store up to 4 qualitative samples for visualization
            if saved_visualizations < 4:
                for b in range(images.shape[0]):
                    if saved_visualizations < 4:
                        # Unnormalize image
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = images[b].cpu().permute(1, 2, 0).numpy()
                        img_np = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
                        
                        gt_mask = targets[b]
                        pred_mask = pred_classes[b]
                        vis_samples.append({
                            'name': names[b],
                            'image': img_np,
                            'gt': gt_mask,
                            'pred': pred_mask
                        })
                        saved_visualizations += 1

    # 7. Aggregate & Report Point 5 Metrics
    metrics_summary = metric_engine.compute_metrics()
    
    print("\n" + "=" * 70)
    print("POINT 5 EVALUATION METRICS REPORT (NIGHTCITY VAL):")
    print("=" * 70)
    print(f"  - Mean IoU (mIoU):              {metrics_summary['mIoU'] * 100:.2f}%")
    print(f"  - Mean Dice Score (F1):         {metrics_summary['mean_Dice'] * 100:.2f}%")
    print(f"  - Mean Sensitivity (Recall):    {metrics_summary['mean_Sensitivity'] * 100:.2f}%")
    print(f"  - Mean Specificity:             {metrics_summary['mean_Specificity'] * 100:.2f}%")
    print(f"  - Pixel Accuracy:               {metrics_summary['Pixel_Accuracy'] * 100:.2f}%")
    print(f"  - Total Valid Pixels:           {metrics_summary['Total_Pixels_Evaluated']:,}")

    # Per-Class Metrics DataFrame
    df_per_class = pd.DataFrame(metrics_summary['Per_Class'])
    # Format percentages for presentation
    df_per_class['IoU (%)'] = df_per_class['IoU'] * 100
    df_per_class['Dice (%)'] = df_per_class['Dice'] * 100
    df_per_class['Sensitivity (%)'] = df_per_class['Sensitivity'] * 100
    df_per_class['Specificity (%)'] = df_per_class['Specificity'] * 100
    
    csv_path = os.path.join(output_dir, "nightcity_per_class_metrics.csv")
    df_per_class.to_csv(csv_path, index=False)
    print(f"\nPer-Class detailed metrics saved to: {csv_path}")

    # 8. Save Full Evaluation JSON
    results_json = {
        'model': 'PIDNet-S',
        'dataset': 'NightCity',
        'split': 'val',
        'samples_evaluated': eval_count,
        'hardware': hw_info,
        'complexity': {
            'parameters': param_info,
            'flops_profile': flop_results
        },
        'inference_speed': speed_results,
        'metrics': {
            'mIoU': metrics_summary['mIoU'],
            'mean_Dice': metrics_summary['mean_Dice'],
            'mean_Sensitivity': metrics_summary['mean_Sensitivity'],
            'mean_Specificity': metrics_summary['mean_Specificity'],
            'pixel_accuracy': metrics_summary['Pixel_Accuracy'],
            'mean_accuracy': metrics_summary['Mean_Accuracy'],
            'total_pixels': metrics_summary['Total_Pixels_Evaluated']
        },
        'per_class': metrics_summary['Per_Class']
    }
    
    json_path = os.path.join(output_dir, "nightcity_baseline_eval.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)
    print(f"Comprehensive metrics JSON saved to: {json_path}")

    # 9. Qualitative Visual Inspection Figure
    if vis_samples:
        print("\nGENERATING QUALITATIVE VISUALIZATION FIGURE...")
        fig, axes = plt.subplots(len(vis_samples), 4, figsize=(20, 4.5 * len(vis_samples)))
        if len(vis_samples) == 1:
            axes = np.expand_dims(axes, 0)

        for i, sample in enumerate(vis_samples):
            img = sample['image']
            gt_rgb = colorize_mask(sample['gt'])
            pred_rgb = colorize_mask(sample['pred'])
            overlay = (0.55 * img + 0.45 * pred_rgb).astype(np.uint8)

            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Night Image: {sample['name']}", fontsize=11, fontweight='bold')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gt_rgb)
            axes[i, 1].set_title("Ground Truth Mask", fontsize=11, fontweight='bold')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title("PIDNet-S Prediction", fontsize=11, fontweight='bold')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title("Overlay (Pred on Image)", fontsize=11, fontweight='bold')
            axes[i, 3].axis('off')

        plt.tight_layout()
        vis_path = os.path.join(output_dir, "nightcity_baseline_predictions.png")
        plt.savefig(vis_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  [INFO] Saved qualitative predictions figure to: {vis_path}")

    print("\n" + "=" * 70)
    print("[INFO] TASK 3 BASELINE EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return results_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate PIDNet-S on NightCity")
    parser.add_argument('--data_root', type=str, default='AI512_Project/Data')
    parser.add_argument('--list_path', type=str, default='PIDNet/data/list/nightcity/val.lst')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=None, help='Evaluate subset of samples for quick profiling')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='output/task3_baseline')
    args = parser.parse_args()

    run_evaluation(
        data_root=args.data_root,
        list_path=args.list_path,
        weights_path=args.weights,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
