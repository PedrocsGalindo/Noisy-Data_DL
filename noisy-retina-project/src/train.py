"""Minimal CE/SCE/QMix-like training entrypoint for Kaggle DR experiments."""

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.kaggle_dr import build_dataloaders
from src.methods.ce import CrossEntropyMethod
from src.methods.qmix_like import QMixLikeHelper
from src.models.resnet import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import accuracy
from src.utils.seed import set_seed


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def save_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

def test(model, test_loader):
    pass

def main():
    parser = argparse.ArgumentParser(
        description="Train CE/SCE/QMix-like Kaggle DR model."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders(config, device)

    model_config = config.get("model", {})
    output_dir = Path(config.get("output", {}).get("dir", config.get("output_dir", "outputs/run")))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        backbone=model_config.get("backbone", "resnet18"),
        num_classes=int(model_config.get("num_classes", 5)),
        pretrained=bool(model_config.get("pretrained", True)),
    ).to(device)

    if config.method == "ce":
        method = CrossEntropyMethod(config, model, device)

    model, results = method.train(train_loader, val_loader)
    test(model, test_loader)


if __name__ == "__main__":
    main()
