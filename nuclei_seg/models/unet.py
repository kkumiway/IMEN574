"""
U-Net with pretrained encoder via segmentation_models_pytorch.

Mirrors the encoder-decoder + FPN design in selim's unets.py but uses
modern smp instead of hand-rolled Keras ResNet / Xception builds.

Default: ResNet-34 encoder, 2-channel sigmoid output (body + border).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def make_model(
    encoder: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 2,
) -> nn.Module:
    """Return a U-Net with FPN decoder and sigmoid activation."""
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation="sigmoid",
    )
    return model


def load_model(
    weights_path: str,
    encoder: str = "resnet34",
    device: str = "cpu",
) -> nn.Module:
    model = make_model(encoder=encoder)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model