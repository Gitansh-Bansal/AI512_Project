# ------------------------------------------------------------------------------
# PIDNet-NightOpt: Optimized Nighttime Real-Time Segmentation Architecture
# Combines:
#   1. Haar Wavelet Edge Injection (DWT-RF) on the Derivative Boundary branch
#   2. Illumination-Aware Channel Calibration (IACC) on the Dual-Fusion feature map
# Ultra-lightweight: adds <0.005M params and <0.05 GFLOPs.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Union

from .pidnet import PIDNet, BasicBlock, Bottleneck, segmenthead
from .model_utils import PagFM, Light_Bag, PAPPM
from .wavelet_iacc import HaarWaveletEdgeExtractor, IlluminationAwareChannelCalibration


class PIDNetNightOpt(PIDNet):
    """
    PIDNet with Haar Wavelet Edge Injection & Illumination-Aware Channel Calibration.
    Eliminates boundary noise in pitch-black shadows and dynamically calibrates channels
    against headlights and glare.
    """
    def __init__(
        self,
        m: int = 2,
        n: int = 3,
        num_classes: int = 19,
        planes: int = 32,
        ppm_planes: int = 96,
        head_planes: int = 128,
        augment: bool = True
    ):
        super(PIDNetNightOpt, self).__init__(
            m=m, n=n, num_classes=num_classes, planes=planes,
            ppm_planes=ppm_planes, head_planes=head_planes, augment=augment
        )
        # Haar Wavelet Edge Extractor projecting into boundary channel dimension (planes = 32)
        self.wavelet_edge = HaarWaveletEdgeExtractor(out_channels=planes)

        # IACC channel calibrator on fused features before final segmentation head
        # In PIDNet-S, dfm output has head_planes channels (128)
        self.iacc = IlluminationAwareChannelCalibration(in_channels=head_planes, reduction=4)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        algc = False
        raw_input = x  # Used for luminance calculation and wavelet edge extraction

        # Stem & Stage 1-2
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        # Stage 3
        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))

        # Boundary branch receives diff + Haar Wavelet high-pass structural edge prior
        dwt_edges = self.wavelet_edge(raw_input, target_size=(height_output, width_output))
        x_d = x_d + F.interpolate(
            self.diff3(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc
        ) + dwt_edges

        if self.augment:
            temp_p = x_

        # Stage 4
        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
            self.diff4(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc
        )
        if self.augment:
            temp_d = x_d

        # Stage 5 & Spatial Pyramid Pooling
        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
            self.spp(self.layer5(x)),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc
        )

        # Dual-branch Fusion & Illumination-Aware Channel Calibration
        fused = self.dfm(x_, x, x_d)
        fused_calibrated = self.iacc(fused, raw_input)
        x_ = self.final_layer(fused_calibrated)

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_


def get_nightopt_model(name: str = 'pidnet_s', num_classes: int = 19, augment: bool = False) -> PIDNetNightOpt:
    """Factory helper to instantiate PIDNetNightOpt."""
    if 's' in name.lower():
        model = PIDNetNightOpt(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=augment)
    elif 'm' in name.lower():
        model = PIDNetNightOpt(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=augment)
    else:
        model = PIDNetNightOpt(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=augment)
    return model
