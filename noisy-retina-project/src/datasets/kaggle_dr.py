"""Dataset and dataloader helpers for Kaggle Diabetic Retinopathy."""

from collections import Counter
from pathlib import Path
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.routes import DATA_PATH

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
        labels,
        images,
        config,
        train= False,
    ):
        self.labels = labels
        self.images = images
        self.image_size = config.get("image_size", 224)
        max_samples = config.get("max_samples", False)
        self.image_dir = config.image_dir

        if max_samples is not None:
            self.labels = self.labels[:max_samples]
            self.images = self.images[:max_samples]
        if train:
            self.transform = build_train_transforms(self.image_size)
        else:
            self.transform = build_val_transforms(self.image_size)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.labels[index]
        filename = self.image_names[index] + ".jpeg"
        image_path = DATA_PATH/ self.image_dir / filename
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
        elif self.allow_missing_images:
            image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
        else:
            raise FileNotFoundError(f"Missing image file: {image_path}")

        return {
            "image": self.transform(image),
            "label": torch.tensor(int(label), dtype=torch.long),
            "index": torch.tensor(int(index), dtype=torch.long),
            "image_name": filename,
        }

def build_train_transforms(imafe_size):
    return transforms.Compose([
        transforms.Resize((int(imafe_size), int(imafe_size))),  # 
        transforms.RandomResizedCrop(size=512, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),  # 
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
        transforms.RandomRotation(degrees=5),  #  ±5°
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])

def build_val_transforms(image_size):
    """Create basic ImageNet-style transforms."""
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

# MAIN func 
def build_dataloaders(config):
    """Build train and validation dataloaders from a YAML config dict."""
    mode = config.get("name", "dummy")
    seed = config.get("seed", 42)
    num_classes = config.get("num_classes", config.get("num_classes", 5))
    val_size = config.get("val_size", 0.1)
    test_size = config.get("test_size", 0.3)
    num_classes= config.num_classes
    batch_size = config.get("batch_size", 16 )

    csv_path = getattr(config, "csv_path", "")
    metadata = pd.read_csv(DATA_PATH/ csv_path)
    labels = metadata["level"].astype(int).tolist()
    images_name = metadata["image"]
    X_train, X_test, y_train, y_test = train_test_split(
        images_name,
        labels,
        test_size=test_size,
        random_state=seed
    )
    val_size = int(len(X_train) * val_size) 
    if mode == "dummy":
        dataset = DummyRetinaDataset(
            num_samples=config.get("num_samples", 24),
            num_classes=num_classes,
            image_size=image_size,
        )
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

    elif mode == "kaggle_dr":
        train_dataset = KaggleDRDataset(
            images=X_train[:val_size],
            labels=y_train[:val_size],
            config=config,
            train=True
        )
        val_dataset = KaggleDRDataset(
            images=X_train[val_size:],
            labels=y_train[val_size:],
            config=config,
            train=False
        )
        test_dataset = KaggleDRDataset(
            images=X_test,
            labels=y_test,
            config=config,
            train=False
        )
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader