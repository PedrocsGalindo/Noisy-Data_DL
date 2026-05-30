"""Dataset and dataloader helpers for Kaggle Diabetic Retinopathy."""

from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class DummyRetinaDataset(Dataset):
    """Synthetic retina-like tensors for local smoke tests."""

    def __init__(self, num_samples=24, num_classes=5, image_size=64):
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.labels = [idx % self.num_classes for idx in range(self.num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(int(index))
        image = torch.rand(
            3,
            self.image_size,
            self.image_size,
            generator=generator,
            dtype=torch.float32,
        )
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


class KaggleDRDataset(Dataset):
    """Kaggle DR image dataset backed by trainLabels.csv."""

    def __init__(
        self,
        csv_path,
        image_dir,
        image_size=224,
        max_samples=None,
        allow_missing_images=False,
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.allow_missing_images = bool(allow_missing_images)
        self.df = pd.read_csv(self.csv_path)

        if "image" not in self.df.columns:
            raise ValueError("Missing required CSV column: image")
        if "level" not in self.df.columns:
            raise ValueError("Missing required CSV column: level")

        if max_samples is not None:
            self.df = self.df.head(int(max_samples))

        self.df = self.df.reset_index(drop=True)
        self.labels = self.df["level"].astype(int).tolist()
        self.transform = build_transforms(self.image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = str(row["image"])
        image_path = self._resolve_image_path(image_name)

        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
        elif self.allow_missing_images:
            image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
        else:
            raise FileNotFoundError(f"Missing image file: {image_path}")

        return {
            "image": self.transform(image),
            "label": torch.tensor(int(row["level"]), dtype=torch.long),
            "index": torch.tensor(int(index), dtype=torch.long),
            "image_name": image_name,
        }

    def _resolve_image_path(self, image_name):
        image_path = Path(image_name)
        if image_path.suffix:
            return self.image_dir / image_path
        return self.image_dir / f"{image_name}.jpeg"


def build_transforms(image_size):
    """Create basic ImageNet-style transforms."""
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _split_indices(labels, val_split, seed):
    indices = list(range(len(labels)))
    if len(indices) < 2 or float(val_split) <= 0.0:
        return indices, []

    class_counts = Counter(labels)
    can_stratify = len(class_counts) > 1 and min(class_counts.values()) >= 2
    stratify = labels if can_stratify else None

    try:
        return train_test_split(
            indices,
            test_size=float(val_split),
            random_state=int(seed),
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            indices,
            test_size=float(val_split),
            random_state=int(seed),
            stratify=None,
        )


def build_dataloaders(config):
    """Build train and validation dataloaders from a YAML config dict."""
    dataset_config = config.get("dataset", {})
    dataloader_config = config.get("dataloader", {})
    train_config = config.get("train", {})
    mode = dataset_config.get("name", dataset_config.get("mode", "dummy"))
    seed = config.get("seed", 42)
    model_config = config.get("model", {})
    num_classes = model_config.get("num_classes", config.get("num_classes", 5))
    image_size = dataset_config.get("image_size", 224)
    val_split = train_config.get("val_split", dataset_config.get("val_split", 0.1))

    if mode == "dummy":
        dataset = DummyRetinaDataset(
            num_samples=dataset_config.get("num_samples", 24),
            num_classes=num_classes,
            image_size=image_size,
        )
        train_idx, val_idx = _split_indices(dataset.labels, val_split, seed)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
    elif mode == "kaggle_dr":
        dataset = KaggleDRDataset(
            csv_path=dataset_config["csv_path"],
            image_dir=dataset_config["image_dir"],
            image_size=image_size,
            max_samples=dataset_config.get("max_samples"),
            allow_missing_images=dataset_config.get("allow_missing_images", False),
        )
        train_idx, val_idx = _split_indices(dataset.labels, val_split, seed)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

    loader_kwargs = {
        "batch_size": int(dataloader_config.get("batch_size", train_config.get("batch_size", 32))),
        "num_workers": int(dataloader_config.get("num_workers", 0)),
        "pin_memory": bool(dataloader_config.get("pin_memory", False)),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
