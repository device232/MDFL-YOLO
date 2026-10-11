"""Equations (8)–(12), integrated without changing assignment, BCE, or DFL."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss


class SlenderAxisLoss(nn.Module):
    """Accept decoded xyxy boxes in the same units and detached soft class scores."""

    def forward(self, pred_boxes, target_boxes, target_scores, foreground):
        if not foreground.any():
            return pred_boxes.sum() * 0.0
        # Evaluate geometry in FP32 even when network outputs use AMP.
        pred = pred_boxes[foreground].float()
        target = target_boxes.detach()[foreground].float()
        size_p = (pred[:, 2:] - pred[:, :2]).clamp_min(1e-6)
        size_t = (target[:, 2:] - target[:, :2]).clamp_min(1e-6)
        center_p = (pred[:, 2:] + pred[:, :2]) * 0.5
        center_t = (target[:, 2:] + target[:, :2]) * 0.5
        aspect = torch.log(size_t[:, 0] / size_t[:, 1])
        horizontal = torch.sigmoid(aspect / 0.5)
        elongation = torch.sigmoid(2.0 * (aspect.abs() - math.log(5.0)))
        center_error = (center_p - center_t) / size_t
        scale_error = torch.log(size_p / size_t)
        penalties = F.smooth_l1_loss(
            center_error, torch.zeros_like(center_error), beta=0.5, reduction="none"
        ) + F.smooth_l1_loss(
            scale_error, torch.zeros_like(scale_error), beta=0.5, reduction="none"
        )
        short_axis = horizontal * penalties[:, 1] + (1.0 - horizontal) * penalties[:, 0]
        scores = target_scores.detach().float()
        quality = scores.sum(dim=-1)[foreground]
        normalizer = scores.sum().clamp_min(1.0)
        return (quality * elongation * short_axis).sum() / normalizer


class SlenderBboxLoss(BboxLoss):
    def __init__(self, reg_max=16, gain=0.1):
        super().__init__(reg_max)
        self.slender = SlenderAxisLoss()
        self.gain = gain

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        iou, dfl = super().forward(
            pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask,
        )
        slender = self.slender(pred_bboxes, target_bboxes, target_scores, fg_mask)
        return iou + self.gain * slender, dfl


class MDFLDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bbox_loss = SlenderBboxLoss(self.reg_max, gain=0.1).to(self.device)
