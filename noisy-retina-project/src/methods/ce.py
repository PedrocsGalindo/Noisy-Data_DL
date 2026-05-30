"""Cross Entropy loss helpers."""

import torch.nn as nn


class CELoss(nn.Module):
    """Thin wrapper around torch.nn.CrossEntropyLoss."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


class CrossEntropyMethod(nn.Module):
    """Compatibility wrapper for older method-based training code."""

    def __init__(self, config=None):
        super().__init__()
        self.loss_fn = CELoss()

    def compute_loss(self, model, images, targets):
        logits = model(images)
        loss = self.loss_fn(logits, targets)
        return loss, logits

    def validation_loss(self, logits, targets):
        return self.loss_fn(logits, targets)
