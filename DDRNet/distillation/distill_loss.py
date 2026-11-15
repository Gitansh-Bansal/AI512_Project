# ------------------------------------------------------------------------------
# Knowledge Distillation Losses for DDRNet-23-slim
# Implements:
#   1. Logit-based Knowledge Distillation (KL Divergence with Temperature T)
#   2. Valid Pixel Masking (prevents transferring gradients onto ignore_label pixels)
#   3. DDRNet Distillation Wrapper (DualResNet Student + Teacher SegFormer-B2)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class LogitKDLoss(nn.Module):
    """
    Logit-based Knowledge Distillation using Kullback-Leibler (KL) Divergence.
    Softens probabilities with temperature T to transfer dark-region class relationships
    and dampen the impact of noisy ground-truth labels.
    """
    def __init__(self, temperature: float = 2.0, ignore_label: int = 255):
        super(LogitKDLoss, self).__init__()
        self.temperature = temperature
        self.ignore_label = ignore_label

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        student_logits: (B, C, H, W)
        teacher_logits: (B, C, H, W)
        labels:         (B, H, W)
        """
        t = self.temperature
        log_prob_s = F.log_softmax(student_logits / t, dim=1)
        prob_t = F.softmax(teacher_logits / t, dim=1)

        # Element-wise KL divergence: sum across class channels C
        kl_div = F.kl_div(log_prob_s, prob_t, reduction='none').sum(dim=1)  # shape: (B, H, W)

        # Mask out ignore_label (255) pixels without CPU stalls
        mask = (labels != self.ignore_label).float()
        loss = (kl_div * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        # Scale by T^2 according to Hinton et al.
        return loss * (t * t)


class DDRNetDistillationWrapper(nn.Module):
    """
    Combines Student DDRNet-23-slim with Teacher SegFormer-B2.
    Evaluates Student semantic loss (Main Head + 0.4 * Aux Head) and KD loss with Teacher.
    """
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        sem_loss: nn.Module,
        kd_loss: nn.Module,
        aux_weight: float = 0.4,
        kd_weight: float = 0.2
    ):
        super(DDRNetDistillationWrapper, self).__init__()
        self.student = student
        self.teacher = teacher
        self.sem_loss = sem_loss
        self.kd_loss = kd_loss
        self.aux_weight = aux_weight
        self.kd_weight = kd_weight

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        h, w = labels.size(1), labels.size(2)

        # 1. Student forward pass
        student_outs = self.student(inputs)
        if isinstance(student_outs, (list, tuple)):
            pred_main, pred_aux = student_outs[0], student_outs[1]
            if pred_main.shape[-2:] != (h, w):
                pred_main = F.interpolate(pred_main, size=(h, w), mode='bilinear', align_corners=True)
            if pred_aux.shape[-2:] != (h, w):
                pred_aux = F.interpolate(pred_aux, size=(h, w), mode='bilinear', align_corners=True)
            loss_main = self.sem_loss(pred_main, labels)
            loss_aux = self.sem_loss(pred_aux, labels)
            loss_sem = loss_main + self.aux_weight * loss_aux
        else:
            pred_main = student_outs
            if pred_main.shape[-2:] != (h, w):
                pred_main = F.interpolate(pred_main, size=(h, w), mode='bilinear', align_corners=True)
            loss_sem = self.sem_loss(pred_main, labels)

        # 2. Frozen teacher forward pass (no gradients needed)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs, target_size=(h, w))

        # 3. Compute KD loss between main student logits and teacher logits
        loss_kd = self.kd_loss(pred_main, teacher_logits, labels)

        total_loss = loss_sem + self.kd_weight * loss_kd
        return total_loss, loss_sem, loss_kd, pred_main
