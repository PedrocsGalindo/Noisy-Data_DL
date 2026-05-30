"""Checkpoint helpers."""

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    """Save model, optimizer, epoch, metrics, and config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
        },
        path,
    )
