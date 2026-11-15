# ------------------------------------------------------------------------------
# Haar Wavelet Transform & Illumination-Aware Channel Calibration (IACC)
# Specifically designed to solve the two biggest failure points on NightCity:
#   1. Haar Wavelet Edge Injection (DWT-RF): Invariant to smooth lighting drift,
#      extracts true high-frequency edge gradients even in near-zero contrast.
#   2. Illumination-Aware Channel Calibration (IACC): Ultra-lightweight (<3K params)
#      SE-style gating conditioned on global scene luminance.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletEdgeExtractor(nn.Module):
    """
    Fixed 2D Haar Wavelet Decomposition.
    Decomposes input luminance into:
      - LL: Low-frequency illumination (smooth lighting / headlight drift)
      - LH, HL, HH: High-frequency structural boundaries (horizontal, vertical, diagonal edges)
    Filters out the smooth low-frequency lighting and returns sharp edge features.
    Computation is pure additions and subtractions (zero learnable params, ultra-fast).
    """
    def __init__(self, out_channels: int = 32):
        super(HaarWaveletEdgeExtractor, self).__init__()
        # Fixed Haar filters (normalized)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)

        self.register_buffer("weight_high", torch.cat([lh, hl, hh], dim=0))  # (3, 1, 2, 2)
        self.proj = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_image: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        rgb_image: (B, 3, H, W)
        target_size: (target_H, target_W) matching boundary branch resolution
        """
        # Compute grayscale luminance
        gray = 0.299 * rgb_image[:, 0:1, :, :] + 0.587 * rgb_image[:, 1:2, :, :] + 0.114 * rgb_image[:, 2:3, :, :]
        # 2D Haar high-pass filtering with stride 2
        high_freq = F.conv2d(gray, self.weight_high, stride=2, padding=0)  # (B, 3, H/2, W/2)
        high_freq = torch.abs(high_freq)

        if high_freq.shape[-2:] != target_size:
            high_freq = F.interpolate(high_freq, size=target_size, mode="bilinear", align_corners=False)

        edge_features = self.proj(high_freq)
        return edge_features


class IlluminationAwareChannelCalibration(nn.Module):
    """
    Illumination-Aware Channel Calibration (IACC).
    Gated Squeeze-and-Excitation conditioned on both feature statistics and scene luminance.
    Dynamically suppresses glare-corrupted channels and boosts underexposed feature maps.
    Parameter overhead: ~2,500 parameters (<0.003M).
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super(IlluminationAwareChannelCalibration, self).__init__()
        reduced = max(8, in_channels // reduction)
        # in_channels + 1 (for global luminance scalar)
        self.fc = nn.Sequential(
            nn.Linear(in_channels + 1, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor, rgb_image: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = feat.shape
        # Global average pool feature channels
        gap_feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(b, c)

        # Global average scene luminance
        lum = 0.299 * rgb_image[:, 0:1, :, :] + 0.587 * rgb_image[:, 1:2, :, :] + 0.114 * rgb_image[:, 2:3, :, :]
        gap_lum = F.adaptive_avg_pool2d(lum, (1, 1)).view(b, 1)

        # Joint descriptor [feat_gap, lum_gap]
        joint = torch.cat([gap_feat, gap_lum], dim=1)
        scale = self.fc(joint).view(b, c, 1, 1)

        # Calibrated residual scaling
        return feat * scale
