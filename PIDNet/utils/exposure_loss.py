# ------------------------------------------------------------------------------
# Exposure-Aware Loss & Self-Supervised Illumination Supervision for PIDNet
# Addresses:
#   1. Blackout Collapse: Upweights pixels in underexposed / pitch-black zones (Y < 0.30)
#   2. Specular Glare / Headlight Bloom: Upweights pixels in saturated glare zones (Y > 0.75)
#   3. Auxiliary Illumination Target Generation (Self-supervised luminance bins)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def compute_normalized_luminance(images: torch.Tensor) -> torch.Tensor:
    """
    Computes perceptual luminance Y in [0, 1] from ImageNet-normalized RGB tensors.
    images shape: (B, 3, H, W)
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)

    # Denormalize to [0, 1]
    rgb = torch.clamp(images * std + mean, 0.0, 1.0)
    # ITU-R BT.601 perceptual luminance formula
    luminance = 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]
    return luminance.squeeze(1)  # shape: (B, H, W)


def compute_illumination_bins(images: torch.Tensor, dark_thresh: float = 0.25, glare_thresh: float = 0.75) -> torch.Tensor:
    """
    Self-supervised 3-bin illumination segmentation target:
      0: Dark / Underexposed (Y < 0.25)
      1: Balanced / Normal    (0.25 <= Y <= 0.75)
      2: Glare / Saturated    (Y > 0.75)
    Returns: (B, H, W) torch.long tensor
    """
    lum = compute_normalized_luminance(images)
    bins = torch.ones_like(lum, dtype=torch.long)
    bins[lum < dark_thresh] = 0
    bins[lum > glare_thresh] = 2
    return bins


def get_exposure_weights(
    luminance: torch.Tensor,
    alpha_dark: float = 1.5,
    beta_glare: float = 1.2,
    tau_dark: float = 0.35,
    tau_glare: float = 0.70
) -> torch.Tensor:
    """
    Calculates pixel-level exposure importance weights W_exp in [1.0, 1.0 + max(alpha, beta)].
    Well-lit midtones receive W_exp = 1.0.
    Under-exposed regions receive up to 1.0 + alpha_dark penalty.
    Over-exposed glare regions receive up to 1.0 + beta_glare penalty.
    """
    weights = torch.ones_like(luminance)

    # Dark penalty
    dark_mask = luminance < tau_dark
    dark_penalty = ((tau_dark - luminance[dark_mask]) / tau_dark) ** 2
    weights[dark_mask] = 1.0 + alpha_dark * dark_penalty

    # Glare penalty
    glare_mask = luminance > tau_glare
    glare_penalty = ((luminance[glare_mask] - tau_glare) / (1.0 - tau_glare)) ** 2
    weights[glare_mask] = 1.0 + beta_glare * glare_penalty

    return weights


class ExposureAwareOhemCrossEntropy(nn.Module):
    """
    Online Hard Example Mining (OHEM) Cross-Entropy weighted by pixel-level exposure difficulty.
    Keeps gradients focused on hard-to-segment low-contrast shadows and specular headlight blooms.
    """
    def __init__(
        self,
        ignore_label: int = 255,
        thres: float = 0.9,
        min_kept: int = 131072,
        class_weight: Optional[torch.Tensor] = None,
        alpha_dark: float = 1.5,
        beta_glare: float = 1.2,
        balance_weights: Tuple[float, float] = (0.4, 1.0)
    ):
        super(ExposureAwareOhemCrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.alpha_dark = alpha_dark
        self.beta_glare = beta_glare
        self.balance_weights = balance_weights

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ohem_forward(self, score: torch.Tensor, target: torch.Tensor, exp_weights: torch.Tensor) -> torch.Tensor:
        # score: (B, C, H, W), target: (B, H, W), exp_weights: (B, H, W)
        pred = F.softmax(score, dim=1)
        raw_losses = self.criterion(score, target)  # (B, H, W)
        # Apply exposure importance weights
        weighted_losses = (raw_losses * exp_weights).contiguous().view(-1)

        mask = (target.contiguous().view(-1) != self.ignore_label)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0

        pred = pred.gather(1, tmp_target.unsqueeze(1)).contiguous().view(-1)[mask]
        sorted_pred, ind = pred.contiguous().sort()

        min_value = sorted_pred[min(self.min_kept, sorted_pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = weighted_losses[mask][ind]
        hard_losses = pixel_losses[sorted_pred < threshold]
        return hard_losses.mean() if hard_losses.numel() > 0 else pixel_losses.mean()

    def _standard_forward(self, score: torch.Tensor, target: torch.Tensor, exp_weights: torch.Tensor) -> torch.Tensor:
        raw_losses = self.criterion(score, target)
        weighted_losses = raw_losses * exp_weights
        mask = (target != self.ignore_label)
        return weighted_losses[mask].mean() if mask.sum() > 0 else weighted_losses.mean()

    def forward(self, score, target: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        score: single tensor or list of [detail_head, main_head]
        target: (B, H, W) ground-truth labels
        images: (B, 3, H, W) input image tensors
        """
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        lum = compute_normalized_luminance(images)
        exp_weights = get_exposure_weights(lum, self.alpha_dark, self.beta_glare)

        if len(score) == 2:
            loss_detail = self._standard_forward(score[0], target, exp_weights)
            loss_main = self._ohem_forward(score[1], target, exp_weights)
            return self.balance_weights[0] * loss_detail + self.balance_weights[1] * loss_main
        else:
            return self._ohem_forward(score[0], target, exp_weights)
