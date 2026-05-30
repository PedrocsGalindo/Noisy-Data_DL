"""Symmetric Cross Entropy loss helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCELoss(nn.Module):
    """Symmetric Cross Entropy: alpha * CE + beta * reverse CE."""

    def __init__(
        self,
        num_classes=5,
        alpha=1.0,
        beta=1.0,
        pred_min=1e-7,
        label_min=1e-4,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pred_min = float(pred_min)
        self.label_min = float(label_min)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)

        pred = torch.softmax(logits, dim=1)
        pred = torch.clamp(pred, min=self.pred_min, max=1.0)

        label_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=self.label_min, max=1.0)

        rce = -torch.sum(pred * torch.log(label_one_hot), dim=1)
        rce = rce.mean()

        return self.alpha * ce + self.beta * rce


class SCEMethod(nn.Module):
    """Compatibility wrapper for older method-based training code."""

    def __init__(self, config):
        super().__init__()
        method_config = config.get("method", {})
        train_config = config.get("train", {})
        model_config = config.get("model", {})
        num_classes = model_config.get("num_classes", config.get("num_classes", 5))
        self.loss_fn = SCELoss(
            num_classes=num_classes,
            alpha=train_config.get("sce_alpha", method_config.get("alpha", 1.0)),
            beta=train_config.get("sce_beta", method_config.get("beta", 1.0)),
        )

    def compute_loss(self, model, images, targets):
        logits = model(images)
        loss = self.loss_fn(logits, targets)
        return loss, logits

    def validation_loss(self, logits, targets):
        return self.loss_fn(logits, targets)
