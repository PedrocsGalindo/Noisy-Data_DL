"""Checkpoint helpers."""

from pathlib import Path

import torch


def save_checkpoint(save_path, model, optimizer, epoch, loss, acc, config):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "loss": float(loss),
            "acc": float(acc),
            "config": config,
        },
        save_path,
    )