"""
Loss functions — mirrors selim's losses.py but in PyTorch.

double_head_loss: 0.5 * BCE-Dice(body) + 0.5 * BCE-Dice(border)
This is the default used for 2-channel (body + border) predictions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce * bce_weight + dice * (1.0 - bce_weight)


def double_head_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, 2, H, W)
      channel 0 = body
      channel 1 = border
    """
    body_loss   = bce_dice_loss(pred[:, 0], target[:, 0])
    border_loss = bce_dice_loss(pred[:, 1], target[:, 1])
    return body_loss + border_loss


LOSSES = {
    "double_head": double_head_loss,
    "bce_dice":    lambda p, t: bce_dice_loss(p[:, 0], t[:, 0]),
}


def make_loss(name: str = "double_head"):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSSES)}")
    return LOSSES[name]