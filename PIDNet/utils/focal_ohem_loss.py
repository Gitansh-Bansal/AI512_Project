# ------------------------------------------------------------------------------
# Robust Focal-OHEM Loss for Nighttime Semantic Segmentation
# Replaces aggressive exposure penalties with Focal Hard-Example Mining
# and label smoothing to prevent gradient explosion in ambiguous dark regions.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class FocalOhemCrossEntropy(nn.Module):
    """
    Focal-OHEM Loss designed for low-light datasets (NightCity).
    Features:
      1. Focal Modulation ((1 - pt)^gamma): Automatically focuses learning on hard
         misclassified boundary pixels without exploding in uninformative pitch black.
      2. Label Smoothing (eps=0.05): Dampens the impact of noisy ground-truth annotations
         in extreme darkness where annotator certainty is low.
      3. OHEM Selection: Retains worst-performing pixels to accelerate convergence.
    """
    def __init__(
        self,
        ignore_label: int = 255,
        gamma: float = 1.5,
        label_smoothing: float = 0.05,
        thres: float = 0.85,
        min_kept: int = 131072,
        class_weight: Optional[torch.Tensor] = None,
        balance_weights: Tuple[float, float] = (0.4, 1.0)
    ):
        super(FocalOhemCrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.balance_weights = balance_weights
        self.class_weight = class_weight

    def _focal_forward(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # score: (B, C, H, W), target: (B, H, W)
        c = score.shape[1]
        mask = (target != self.ignore_label)
        if mask.sum() == 0:
            return score.sum() * 0.0

        # Flatten valid pixels
        score_flat = score.permute(0, 2, 3, 1).contiguous().view(-1, c)[mask.view(-1)]
        target_flat = target.view(-1)[mask.view(-1)]

        # Cross-entropy with label smoothing
        log_prob = F.log_softmax(score_flat, dim=-1)
        prob = torch.exp(log_prob)

        # Gather pt for true classes
        pt = prob.gather(1, target_flat.unsqueeze(1)).squeeze(1)

        # Standard CE loss per pixel
        raw_ce = F.cross_entropy(
            score_flat, target_flat,
            weight=self.class_weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )

        # Focal weighting: (1 - pt)^gamma
        focal_weight = torch.pow(1.0 - pt + 1e-6, self.gamma)
        focal_loss = focal_weight * raw_ce

        # OHEM thresholding without CPU synchronization stalls
        if pt.numel() > self.min_kept:
            # torch.kthvalue is O(N) instead of O(N log N) sort, keeps computation on GPU
            min_pt = torch.kthvalue(pt, self.min_kept)[0]
            thresh_tensor = torch.maximum(min_pt, torch.tensor(self.thresh, device=pt.device, dtype=pt.dtype))
            hard_mask = pt < thresh_tensor
            loss_kept = focal_loss * hard_mask.float()
            num_kept = torch.clamp(hard_mask.sum(), min=1.0)
            return loss_kept.sum() / num_kept
        else:
            return focal_loss.mean()

    def forward(self, score, target: torch.Tensor) -> torch.Tensor:
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        if len(score) == 2:
            loss_detail = self._focal_forward(score[0], target)
            loss_main = self._focal_forward(score[1], target)
            return self.balance_weights[0] * loss_detail + self.balance_weights[1] * loss_main
        else:
            return self._focal_forward(score[0], target)
