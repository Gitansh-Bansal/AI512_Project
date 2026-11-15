# ------------------------------------------------------------------------------
# Knowledge Distillation Losses for Semantic Segmentation
# Implements:
#   1. Logit-based Knowledge Distillation (KL Divergence with Temperature T)
#   2. Valid Pixel Masking (prevents transferring gradients onto ignore_label pixels)
#   3. Full Distillation Wrapper (Student PIDNet-S + Teacher SegFormer-B2)
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
    def __init__(self, temperature: float = 4.0, ignore_label: int = 255):
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


class DistillationWrapper(nn.Module):
    """
    Combines Student PIDNet-S with Teacher SegFormer-B2.
    Evaluates Student semantic loss, boundary loss, and KD loss with Teacher.
    """
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        sem_loss: nn.Module,
        bd_loss: nn.Module,
        kd_loss: nn.Module,
        bd_weight: float = 1.0,
        kd_weight: float = 1.0
    ):
        super(DistillationWrapper, self).__init__()
        self.student = student
        self.teacher = teacher
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss
        self.kd_loss = kd_loss
        self.bd_weight = bd_weight
        self.kd_weight = kd_weight

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, bd_gt: torch.Tensor):
        # 1. Student forward pass
        student_outs = self.student(inputs)  # [x_extra_p, x_main, x_extra_d]
        h, w = labels.size(1), labels.size(2)

        for i in range(len(student_outs)):
            if student_outs[i].size(2) != h or student_outs[i].size(3) != w:
                student_outs[i] = F.interpolate(
                    student_outs[i], size=(h, w), mode='bilinear', align_corners=False
                )

        # 2. Frozen teacher forward pass (no gradients needed)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs, target_size=(h, w))

        # 3. Compute loss components
        loss_sem = self.sem_loss(student_outs[:2], labels).mean()
        loss_bd = self.bd_loss(student_outs[2], bd_gt).mean()
        loss_kd = self.kd_loss(student_outs[1], teacher_logits, labels)

        total_loss = loss_sem + self.bd_weight * loss_bd + self.kd_weight * loss_kd
        return total_loss, loss_sem, loss_bd, loss_kd, student_outs[1]
