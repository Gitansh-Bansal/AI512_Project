# ------------------------------------------------------------------------------
# Task 3 Point 5: TwinLiteNetPlus Adaptation for Full Semantic Segmentation (19 classes)
# Adapts multi-task drivable/lane model into 19-class nocturnal segmentation model.
# Incorporates Hierarchical ESPNet Encoder, CAAM Context Module, and Multi-Stage Decoder.
# ------------------------------------------------------------------------------

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from model.model import Encoder, CAAM, ConvBatchnormRelu, UpConvBlock
    from model import config as cfg
except (ImportError, ModuleNotFoundError):
    from .model import Encoder, CAAM, ConvBatchnormRelu, UpConvBlock
    from . import config as cfg


class TwinLiteNetPlusSeg19(nn.Module):
    """
    Adapted TwinLiteNetPlus Architecture for Full Semantic Segmentation (19 classes).
    
    Architectural Pipeline:
    1. Hierarchical ESPNet Encoder: Extracts multi-scale representations using Depthwise ESP blocks.
    2. Class Activation Attention Module (CAAM): Exploits inter-class contextual correlations via Graph Convolutions.
    3. Multi-Stage Progressive Decoder: Upsamples features with skip connections (inp1, inp2, input image)
       to recover fine nocturnal spatial details and boundaries.
    4. Unconstrained 1x1 Classification Head: Projects final feature representations into 19 logits
       without output clamping, ensuring optimal convergence under Cross-Entropy / OHEM loss.
    """
    def __init__(self, config: str = 'small', num_classes: int = 19):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        if config not in cfg.sc_ch_dict:
            raise ValueError(f"Unknown config '{config}'. Choose from: {list(cfg.sc_ch_dict.keys())}")

        model_cfg = cfg.sc_ch_dict[config]
        self.encoder = Encoder(config)
        c0, c1, c2 = model_cfg['chanels'][0], model_cfg['chanels'][1], model_cfg['chanels'][2]

        # Context Attention
        self.caam = CAAM(feat_in=c2, num_classes=c2, bin_size=(2, 4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = ConvBatchnormRelu(c2, c1)

        # Multi-stage progressive upsampling
        self.up_1 = UpConvBlock(c1, c0)          # stage 1: 1/8 -> 1/4 with inp2
        self.up_2 = UpConvBlock(c0, 16)          # stage 2: 1/4 -> 1/2 with inp1
        self.up_3 = UpConvBlock(16, 16)          # stage 3: 1/2 -> 1/1 with input image
        self.final_cls = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_encoder, inp1, inp2 = self.encoder(x)
        out_caam = self.caam(out_encoder)
        out_caam = self.conv_caam(out_caam)

        out = self.up_1(out_caam, inp2)
        out = self.up_2(out, inp1)
        out = self.up_3(out, x)
        logits = self.final_cls(out)
        return logits


def get_twinlitenet_plus_seg(config: str = 'small', num_classes: int = 19) -> TwinLiteNetPlusSeg19:
    """Factory function for adapted TwinLiteNetPlus semantic segmentation network."""
    return TwinLiteNetPlusSeg19(config=config, num_classes=num_classes)
