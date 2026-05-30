"""Basic classification metrics."""

import math

import torch
from sklearn.metrics import cohen_kappa_score


def accuracy(logits, targets):
    """Return top-1 accuracy as a float between 0.0 and 1.0."""
    if targets.numel() == 0:
        return 0.0

    predictions = torch.argmax(logits, dim=1)
    return float((predictions == targets).float().mean().item())


def classification_metrics(logits, targets, num_classes=5):
    """Return accuracy and quadratic weighted kappa."""
    predictions = torch.argmax(logits, dim=1).detach().cpu()
    targets = targets.detach().cpu()

    if targets.numel() == 0:
        return {"accuracy": 0.0, "quadratic_weighted_kappa": 0.0}

    accuracy_score = (predictions == targets).float().mean().item()
    qwk = cohen_kappa_score(
        targets.tolist(),
        predictions.tolist(),
        labels=list(range(int(num_classes))),
        weights="quadratic",
    )
    if not math.isfinite(float(qwk)):
        qwk = 0.0

    return {
        "accuracy": float(accuracy_score),
        "quadratic_weighted_kappa": float(qwk),
    }
