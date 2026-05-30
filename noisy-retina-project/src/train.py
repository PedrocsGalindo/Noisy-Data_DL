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

from src.datasets.kaggle_dr import KaggleDRDataset
from src.methods.ce import CELoss
from src.methods.qmix_like import QMixLikeHelper
from src.methods.sce import SCELoss
from src.models.resnet import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import accuracy
from src.utils.seed import set_seed


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(config):
    dataset_config = config.get("dataset", {})
    name = dataset_config.get("name", dataset_config.get("mode", "kaggle_dr"))
    if name != "kaggle_dr":
        raise ValueError(f"Unsupported dataset '{name}'. Expected 'kaggle_dr'.")

    return KaggleDRDataset(
        csv_path=dataset_config["csv_path"],
        image_dir=dataset_config["image_dir"],
        image_size=dataset_config.get("image_size", 224),
        max_samples=dataset_config.get("max_samples"),
        allow_missing_images=dataset_config.get("allow_missing_images", False),
    )


def split_dataset(dataset, val_split, seed):
    if len(dataset) < 2 or float(val_split) <= 0.0:
        return dataset, Subset(dataset, [])

    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_size = int(round(len(indices) * float(val_split)))
    val_size = min(max(val_size, 1), len(indices) - 1)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_dataloaders(config, device):
    train_config = config.get("train", {})
    dataset = build_dataset(config)
    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
        val_split=train_config.get("val_split", 0.2),
        seed=config.get("seed", 42),
    )

    loader_kwargs = {
        "batch_size": int(train_config.get("batch_size", 4)),
        "num_workers": int(train_config.get("num_workers", 0)),
        "pin_memory": bool(train_config.get("pin_memory", device.type == "cuda")),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return dataset, train_loader, val_loader


def build_criterion(config):
    train_config = config.get("train", {})
    model_config = config.get("model", {})
    method = str(train_config.get("method", "ce")).lower()
    num_classes = int(model_config.get("num_classes", 5))

    if method == "ce":
        return CELoss(), method
    if method == "qmix_like":
        return CELoss(), method
    if method == "sce":
        return (
            SCELoss(
                num_classes=num_classes,
                alpha=train_config.get("sce_alpha", 1.0),
                beta=train_config.get("sce_beta", 1.0),
            ),
            method,
        )

    raise ValueError(
        f"Unsupported train.method '{method}'. Expected 'ce', 'sce', or 'qmix_like'."
    )


def build_optimizer(config, model):
    train_config = config.get("train", {})
    return torch.optim.Adam(
        model.parameters(),
        lr=float(train_config.get("lr", 1e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["image"], batch["label"], batch.get("index")
    if len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def train_one_epoch(model, criterion, dataloader, optimizer, device, qmix_helper=None):
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in dataloader:
        images, targets, indices = _unpack_batch(batch)
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if qmix_helper is None:
            loss = criterion(logits, targets)
        else:
            if indices is None:
                raise ValueError("QMix-like training requires sample indices in each batch.")
            loss = qmix_helper.weighted_loss(logits, targets, indices, device)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    if total_items == 0:
        return 0.0
    return total_loss / total_items


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    all_logits = []
    all_targets = []

    for batch in dataloader:
        images, targets, _ = _unpack_batch(batch)
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if not all_targets:
        return 0.0

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return accuracy(logits, targets)


def save_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train a minimal CE/SCE/QMix-like Kaggle DR model."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate dataset loading.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = resolve_device()
    dataset, train_loader, val_loader = build_dataloaders(config, device)

    if args.dry_run:
        batch = next(iter(train_loader))
        images, targets, indices = _unpack_batch(batch)
        print(f"train_batches={len(train_loader)} val_batches={len(val_loader)}")
        print(f"batch_image_shape={tuple(images.shape)}")
        print(f"batch_labels={targets}")
        if indices is not None:
            print(f"batch_indices={indices}")
        return

    model_config = config.get("model", {})
    output_dir = Path(config.get("output", {}).get("dir", config.get("output_dir", "outputs/run")))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        backbone=model_config.get("backbone", model_config.get("name", "resnet18")),
        num_classes=int(model_config.get("num_classes", config.get("num_classes", 5))),
        pretrained=bool(model_config.get("pretrained", True)),
    ).to(device)
    criterion, method = build_criterion(config)
    criterion = criterion.to(device)
    optimizer = build_optimizer(config, model)

    train_config = config.get("train", {})
    epochs = int(train_config.get("epochs", 1))
    warmup_epochs = int(train_config.get("warmup_epochs", 1))
    qmix_helper = None
    qmix_groups_fitted = False
    if method == "qmix_like":
        qmix_helper = QMixLikeHelper(random_state=config.get("seed", 42))

    def fit_qmix_groups_once():
        nonlocal qmix_groups_fitted
        if qmix_helper is None or qmix_groups_fitted:
            return
        indices, scores = qmix_helper.compute_scores(model, train_loader, device)
        qmix_helper.fit_groups(indices, scores, dataset_size=len(dataset))
        qmix_helper.save_group_report(output_dir)
        qmix_groups_fitted = True

    metrics = {
        "config": args.config,
        "method": method,
        "epochs": [],
        "best_val_accuracy": 0.0,
    }

    print(f"device={device}")
    print(f"method={method}")
    print(f"output_dir={output_dir}")

    for epoch in range(1, epochs + 1):
        phase = method
        weighted_helper = None

        if method == "qmix_like":
            if epoch <= warmup_epochs:
                phase = "warmup_ce"
            else:
                fit_qmix_groups_once()
                phase = "qmix_like_weighted"
                weighted_helper = qmix_helper

        train_loss = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            qmix_helper=weighted_helper,
        )

        if method == "qmix_like" and epoch == warmup_epochs:
            fit_qmix_groups_once()

        val_accuracy = validate(model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": float(val_accuracy),
        }
        if method == "qmix_like":
            epoch_metrics["phase"] = phase
        metrics["epochs"].append(epoch_metrics)
        metrics["best_val_accuracy"] = max(metrics["best_val_accuracy"], float(val_accuracy))

        message = (
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )
        if method == "qmix_like":
            message = f"{message} phase={phase}"
        print(message)

    if method == "qmix_like":
        fit_qmix_groups_once()

    save_metrics(output_dir / "metrics.json", metrics)
    save_checkpoint(
        output_dir / "last_checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        metrics=metrics,
        config=config,
    )
    print(f"metrics={output_dir / 'metrics.json'}")
    print(f"checkpoint={output_dir / 'last_checkpoint.pt'}")


if __name__ == "__main__":
    main()
