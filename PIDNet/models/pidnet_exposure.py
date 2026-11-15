# ------------------------------------------------------------------------------
# PIDNet with Auxiliary Self-Supervised Illumination Head
# Extends PIDNet-S with an auxiliary illumination prediction head (dropped at inference)
# Guarantees ZERO inference FLOP / latency penalty compared to vanilla PIDNet-S.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Union

from .pidnet import PIDNet, BasicBlock, Bottleneck, segmenthead
from .model_utils import PagFM, Light_Bag, PAPPM


class PIDNetExposure(PIDNet):
    """
    PIDNet with Auxiliary Illumination Head (Option 1).
    During training (self.augment=True):
      Returns: [x_extra_p, x_main, x_extra_d, illum_logits]
    During inference (self.augment=False):
      Returns: x_main  (Zero overhead! Identical graph and parameters to standard PIDNet)
    """
    def __init__(
        self,
        m: int = 2,
        n: int = 3,
        num_classes: int = 19,
        planes: int = 32,
        ppm_planes: int = 96,
        head_planes: int = 128,
        augment: bool = True,
        num_illum_bins: int = 3
    ):
        super(PIDNetExposure, self).__init__(
            m=m, n=n, num_classes=num_classes, planes=planes,
            ppm_planes=ppm_planes, head_planes=head_planes, augment=augment
        )
        self.num_illum_bins = num_illum_bins

        # Auxiliary illumination head attached to high-resolution detail feature temp_p
        # planes * 2 channels at stage 3 (e.g. 64 channels for PIDNet-S)
        in_channels = planes * 2
        self.illum_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_illum_bins, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        algc = False

        # Stem & Stage 1-2
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        # Stage 3
        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
            self.diff3(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc
        )
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

        # Dual-branch Fusion & Final Segmentation Head
        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            illum_logits = self.illum_head(temp_p)
            return [x_extra_p, x_, x_extra_d, illum_logits]
        else:
            return x_


def get_exposure_model(name: str = 'pidnet_s', num_classes: int = 19, augment: bool = True) -> PIDNetExposure:
    """Factory helper to instantiate PIDNetExposure."""
    if 's' in name.lower():
        model = PIDNetExposure(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=augment)
    elif 'm' in name.lower():
        model = PIDNetExposure(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=augment)
    else:
        model = PIDNetExposure(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=augment)
    return model
