# ------------------------------------------------------------------------------
# Task 3 Point 5 Metric Suite for Semantic Segmentation
# Computes: mIoU, Dice Score, Sensitivity, Specificity, Pixel Accuracy
# ------------------------------------------------------------------------------

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
]

CAMVID_CLASSES = [
    'Bicyclist', 'Building', 'Car', 'Pole', 'Fence',
    'Pedestrian', 'Road', 'Sidewalk', 'SignSymbol', 'Sky', 'Tree'
]

CAMVID_PALETTE = [
    [0, 128, 192], [128, 0, 0], [64, 0, 128], [192, 192, 128],
    [64, 64, 128], [64, 64, 0], [128, 64, 128], [0, 0, 192],
    [192, 128, 128], [128, 128, 128], [128, 128, 0]
]

class SegmentationMetrics:
    """
    Comprehensive metric computation for semantic segmentation:
    - Confusion Matrix accumulation
    - Mean Intersection over Union (mIoU)
    - Dice Score (F1-score)
    - Sensitivity (Recall / TPR)
    - Specificity (TNR)
    - Overall Pixel Accuracy & Class-Mean Accuracy
    """
    def __init__(self, num_classes: int = 19, ignore_label: int = 255, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.class_names = class_names if class_names else CITYSCAPES_CLASSES[:num_classes]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        preds: (B, H, W) or (H, W) array of predicted class integers
        targets: (B, H, W) or (H, W) array of ground truth class integers
        """
        preds = np.asarray(preds).flatten()
        targets = np.asarray(targets).flatten()

        # Mask out ignore_label and invalid pixels
        valid_mask = (targets != self.ignore_label) & (targets >= 0) & (targets < self.num_classes)
        preds_valid = preds[valid_mask]
        targets_valid = targets[valid_mask]

        if len(targets_valid) == 0:
            return

        # Fast 2D bincount confusion matrix
        idx = targets_valid * self.num_classes + preds_valid
        counts = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute_metrics(self) -> Dict[str, object]:
        cm = self.confusion_matrix
        tp = np.diag(cm).astype(np.float64)
        pos = cm.sum(axis=1).astype(np.float64)       # Total ground truth per class (TP + FN)
        res = cm.sum(axis=0).astype(np.float64)       # Total predicted per class (TP + FP)
        total_pixels = cm.sum().astype(np.float64)

        # 1. IoU: TP / (TP + FP + FN)
        union = pos + res - tp
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)

        # 2. Dice: 2 * TP / (2 * TP + FP + FN)
        dice_denom = pos + res
        dice = np.divide(2 * tp, dice_denom, out=np.zeros_like(tp), where=dice_denom > 0)

        # 3. Sensitivity (Recall / TPR): TP / (TP + FN)
        sensitivity = np.divide(tp, pos, out=np.zeros_like(tp), where=pos > 0)

        # 4. Specificity (TNR): TN / (TN + FP)
        # TN for class c = total_pixels - pos[c] - res[c] + tp[c]
        # TN + FP = total_pixels - pos[c]
        fp = res - tp
        tn = total_pixels - pos - res + tp
        spec_denom = tn + fp
        specificity = np.divide(tn, spec_denom, out=np.zeros_like(tn), where=spec_denom > 0)

        # Summary averages (averaged over present/valid classes)
        present_classes = pos > 0
        mean_iou = float(np.mean(iou[present_classes])) if np.any(present_classes) else 0.0
        mean_dice = float(np.mean(dice[present_classes])) if np.any(present_classes) else 0.0
        mean_sens = float(np.mean(sensitivity[present_classes])) if np.any(present_classes) else 0.0
        mean_spec = float(np.mean(specificity[present_classes])) if np.any(present_classes) else 0.0

        pixel_acc = float(tp.sum() / max(1.0, total_pixels))
        mean_acc = mean_sens  # class-averaged accuracy is equivalent to mean sensitivity

        per_class_data = []
        for c in range(self.num_classes):
            per_class_data.append({
                'Class_ID': c,
                'Class_Name': self.class_names[c] if c < len(self.class_names) else f'Class_{c}',
                'Pixels_GT': int(pos[c]),
                'IoU': float(iou[c]),
                'Dice': float(dice[c]),
                'Sensitivity': float(sensitivity[c]),
                'Specificity': float(specificity[c])
            })

        return {
            'mIoU': mean_iou,
            'mean_Dice': mean_dice,
            'mean_Sensitivity': mean_sens,
            'mean_Specificity': mean_spec,
            'Pixel_Accuracy': pixel_acc,
            'Mean_Accuracy': mean_acc,
            'Total_Pixels_Evaluated': int(total_pixels),
            'Per_Class': per_class_data,
            'Confusion_Matrix': cm
        }
