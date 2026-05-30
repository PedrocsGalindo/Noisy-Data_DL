# Loss method factory.

from src.methods.ce import CELoss, CrossEntropyMethod
from src.methods.qmix_like import QMixLikeHelper
from src.methods.sce import SCELoss, SCEMethod


def build_method(config):
    method_config = config.get("method", {})
    train_config = config.get("train", {})
    name = method_config.get("name", train_config.get("method", "ce")).lower()

    if name == "ce":
        return CrossEntropyMethod(config)
    if name == "sce":
        return SCEMethod(config)
    if name == "qmix_like":
        raise ValueError("qmix_like is handled directly in src/train.py with QMixLikeHelper.")

    raise ValueError(f"Unknown method: {name}")
