# ------------------------------------------------------------------------------
# TwinLiteNet+ Evaluation and Inference Pipeline on BDD100K Dataset
# Evaluates Drivable Area mIoU, Lane Line Accuracy & IoU, Params, FLOPs, and FPS
# Supports evaluating official Nano/Small/Medium/Large checkpoints & CPU-trained runs.
# ------------------------------------------------------------------------------

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model.model import TwinLiteNetPlus
from benchmarks import profile_parameters, profile_flops, benchmark_inference_speed
from train_bdd100k import BDD100KMultiTaskDataset, evaluate_bdd100k


def save_bdd100k_visuals(model: torch.nn.Module, dataset: BDD100KMultiTaskDataset, device: torch.device, out_dir: str, num_vis: int = 6):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(min(num_vis, len(dataset))):
            fname, img_t, gt_da, gt_ll = dataset[i]
            out_da, out_ll = model(img_t.unsqueeze(0).to(device))
            pred_da = torch.argmax(out_da, dim=1)[0, 12:-12, :].cpu().numpy()
            pred_ll = torch.argmax(out_ll, dim=1)[0, 12:-12, :].cpu().numpy()

            img_vis = (img_t[:, 12:-12, :].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()
            # Green for Drivable Area, Red for Lane Line
            overlay = img_vis.copy()
            overlay[pred_da == 1] = [0, 200, 80]
            overlay[pred_ll == 1] = [230, 40, 40]
            blended = cv2.addWeighted(img_vis, 0.58, overlay, 0.42, 0)
            cv2.imwrite(os.path.join(out_dir, f"{Path(fname).stem}_overlay.png"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def evaluate_checkpoint(
    data_root: str = "TwinLiteNetPlus/data/bdd100k",
    hyp_path: str = "TwinLiteNetPlus/hyperparameters/twinlitev2_hyper.yaml",
    model_config: str = "small",
    weights_path: Optional[str] = None,
    val_samples: Optional[int] = None,
    batch_size: int = 4,
    save_vis_dir: Optional[str] = None,
) -> Dict[str, object]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(hyp_path, 'r', encoding='utf-8') as f:
        hyp = yaml.safe_load(f)

    val_ds = BDD100KMultiTaskDataset(data_root, split="val", hyp=hyp, valid=True, max_samples=val_samples)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TwinLiteNetPlus(SimpleNamespace(config=model_config)).to(device)
    if weights_path and os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        clean_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=True)

    params = profile_parameters(model)
    flops = profile_flops(model, [(384, 640)], device=device)[0]
    speed = benchmark_inference_speed(model, input_resolution=(384, 640), batch_size=1, device=device, warmup_runs=5, timed_runs=15)
    metrics = evaluate_bdd100k(model, val_loader, device)

    if save_vis_dir:
        save_bdd100k_visuals(model, val_ds, device, save_vis_dir, num_vis=6)

    return {
        'Config': model_config,
        'Samples': len(val_ds),
        'Params_M': round(params['total_params_M'], 3),
        'FLOPs_G': round(flops.get('FLOPs_G', 0.0), 2),
        'CPU_Latency_ms': round(speed['Mean_Latency_ms'], 2),
        'CPU_FPS': round(speed['FPS'], 2),
        **metrics
    }


def eval_all_official(
    data_root: str = "TwinLiteNetPlus/data/bdd100k",
    pretrained_dir: str = "TwinLiteNetPlus/pretrained",
    output_dir: str = "output/bdd100k_twinlitenetplus_official",
    val_samples: Optional[int] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for cfg in ['nano', 'small', 'medium', 'large']:
        ckpt = os.path.join(pretrained_dir, f"{cfg}.pth")
        if not os.path.exists(ckpt):
            print(f"[WARN] Missing checkpoint: {ckpt}")
            continue
        vis_dir = os.path.join(output_dir, f"vis_{cfg}") if cfg in ('small', 'large') else None
        res = evaluate_checkpoint(
            data_root=data_root,
            model_config=cfg,
            weights_path=ckpt,
            val_samples=val_samples,
            save_vis_dir=vis_dir
        )
        rows.append(res)
        print(
            f"[{cfg.upper():6s}] Params={res['Params_M']:.3f}M | FLOPs={res['FLOPs_G']:.2f}G | "
            f"DA mIoU={res['DA_mIoU_percent']:.2f}% | LL Acc={res['LL_Accuracy_percent']:.2f}% | "
            f"LL IoU={res['LL_IoU_percent']:.2f}% | {res['CPU_FPS']:.1f} FPS"
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "official_bdd100k_comparison.csv"), index=False)
    with open(os.path.join(output_dir, "official_bdd100k_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate TwinLiteNet+ on BDD100K")
    parser.add_argument('--data_root', type=str, default='TwinLiteNetPlus/data/bdd100k')
    parser.add_argument('--config', type=str, default='small')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--val_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='output/bdd100k_twinlitenetplus_test')
    parser.add_argument('--eval_all_official', action='store_true')
    args = parser.parse_args()

    if args.eval_all_official:
        eval_all_official(data_root=args.data_root, output_dir=args.output_dir, val_samples=args.val_samples)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        res = evaluate_checkpoint(
            data_root=args.data_root,
            model_config=args.config,
            weights_path=args.weights,
            val_samples=args.val_samples,
            save_vis_dir=os.path.join(args.output_dir, "overlays")
        )
        print(json.dumps(res, indent=2))
        with open(os.path.join(args.output_dir, "test_metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
