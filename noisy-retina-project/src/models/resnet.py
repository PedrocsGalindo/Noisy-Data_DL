"""ResNet model factory."""

from pathlib import Path
from urllib.parse import urlparse
import warnings

import torch
import torch.nn as nn
from torchvision import models


_BACKBONES = {
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
}


def _cached_weights_available(weights):
    if weights is None:
        return False
    filename = Path(urlparse(weights.url).path).name
    return (Path(torch.hub.get_dir()) / "checkpoints" / filename).exists()


def build_model(backbone="resnet18", num_classes=5, pretrained=True):
    """Build a ResNet classifier with the final layer sized for DR classes."""
    backbone = str(backbone).lower()
    if backbone not in _BACKBONES:
        supported = ", ".join(sorted(_BACKBONES))
        raise ValueError(f"Invalid backbone '{backbone}'. Supported backbones: {supported}")

    builder, weights_name = _BACKBONES[backbone]
    weights_cls = getattr(models, weights_name, None)
    weights = weights_cls.DEFAULT if pretrained and weights_cls is not None else None

    if pretrained and weights is not None and not _cached_weights_available(weights):
        warnings.warn(
            f"Pretrained {backbone} weights were requested but are not available in the "
            "local torch cache. Falling back to random initialization.",
            RuntimeWarning,
        )
        weights = None

    try:
        model = builder(weights=weights)
    except TypeError:
        model = builder(pretrained=bool(pretrained and weights is not None))

    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    return model
