# ------------------------------------------------------------------------------
# Teacher Model: SegFormer-B2 (Cityscapes Pretrained / Fine-tuned)
# Encapsulates SegFormer-B2 as a frozen teacher for knowledge distillation into PIDNet-S
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SegFormerTeacher(nn.Module):
    """
    Wraps SegFormer-B2 semantic segmentation model as a teacher.
    Extracts softened class probability distributions for dark-region knowledge transfer.
    """
    def __init__(self, pretrained_model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"):
        super(SegFormerTeacher, self).__init__()
        from transformers import SegformerForSemanticSegmentation
        self.teacher = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, target_size: Optional[tuple] = None) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        returns: (B, 19, target_H, target_W) logits
        """
        if target_size is None:
            target_size = (x.shape[-2], x.shape[-1])

        with torch.no_grad():
            out = self.teacher(x)
            logits = out.logits
            if logits.shape[-2:] != target_size:
                logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits


def get_segformer_teacher(device: torch.device) -> SegFormerTeacher:
    teacher = SegFormerTeacher().to(device)
    teacher.eval()
    return teacher
