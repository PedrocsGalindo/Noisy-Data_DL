"""Minimal LNMBench-facing adapter without importing LNMBench internals."""

from src.datasets.kaggle_dr import build_dataloaders
from src.methods import build_method
from src.models.resnet import build_model


class LNMBenchAdapter:
    """Small boundary object for future LNMBench integration."""

    def __init__(self, config):
        self.config = config

    def get_dataloaders(self):
        return build_dataloaders(self.config)

    def get_model(self):
        model_config = self.config.get("model", {})
        return build_model(
            num_classes=self.config.get("num_classes", 5),
            pretrained=model_config.get("pretrained", False),
        )

    def get_method(self):
        return build_method(self.config)

    def as_dict(self):
        return {
            "config": self.config,
            "build_dataloaders": self.get_dataloaders,
            "build_model": self.get_model,
            "build_method": self.get_method,
        }


def build_lnmbench_interface(config):
    """Return a simple integration surface for external runners."""
    return LNMBenchAdapter(config).as_dict()
