"""QMix-like reliability grouping helper.

This is inspired by QMix-style sample reliability separation. It is not an
implementation of the original QMix method.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


class QMixLikeHelper:
    """Score samples, split them into reliability groups, and apply weights."""

    GROUP_WEIGHTS = {
        "clean": 1.0,
        "medium": 0.5,
        "noisy": 0.2,
    }
    GROUP_ORDER = ("clean", "medium", "noisy")

    def __init__(self, num_groups=3, random_state=42):
        self.num_groups = int(num_groups)
        self.random_state = int(random_state)
        self.sample_weights = None
        self.grouping_strategy = None
        self.group_counts = {group: 0 for group in self.GROUP_ORDER}
        self.group_score_means = {group: 0.0 for group in self.GROUP_ORDER}

    @torch.no_grad()
    def compute_scores(self, model, dataloader, device):
        """Return dataset indices and CE+entropy scores for each sample."""
        was_training = model.training
        model.eval()

        all_indices = []
        all_scores = []

        for batch in dataloader:
            images, labels, indices = self._unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            ce = F.cross_entropy(logits, labels, reduction="none")

            probs = torch.softmax(logits, dim=1)
            probs = torch.clamp(probs, min=1e-7, max=1.0)
            entropy = -torch.sum(probs * torch.log(probs), dim=1)

            scores = ce + entropy
            all_indices.append(indices.detach().cpu().long())
            all_scores.append(scores.detach().cpu().float())

        if was_training:
            model.train()

        if not all_indices:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)

        return torch.cat(all_indices, dim=0), torch.cat(all_scores, dim=0)

    def fit_groups(self, indices, scores, dataset_size):
        """Fit reliability groups and store a per-sample weight vector."""
        indices = torch.as_tensor(indices, dtype=torch.long).cpu()
        scores = torch.as_tensor(scores, dtype=torch.float32).cpu()
        dataset_size = int(dataset_size)
        self.sample_weights = torch.ones(dataset_size, dtype=torch.float32)

        if indices.numel() == 0:
            print("QMix-like grouping: no scores found; keeping unit weights.")
            self.grouping_strategy = "percentile_fallback"
            return

        groups = self._fit_gaussian_mixture(scores)
        if groups is None:
            print("QMix-like grouping: using percentile fallback.")
            groups = self._fit_percentiles(scores)
            self.grouping_strategy = "percentile_fallback"
        else:
            self.grouping_strategy = "gaussian_mixture"

        for sample_index, group in zip(indices.tolist(), groups):
            if 0 <= sample_index < dataset_size:
                self.sample_weights[sample_index] = self.GROUP_WEIGHTS[group]

        self._update_report(groups, scores)

    def weighted_loss(self, logits, labels, indices, device):
        """Return mean CE loss weighted by the fitted sample reliability group."""
        if self.sample_weights is None:
            raise RuntimeError("QMix-like sample weights are not fitted yet.")

        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        indices = torch.as_tensor(indices, dtype=torch.long, device=device)
        weights = self.sample_weights.to(device)[indices]
        return (per_sample_loss * weights).mean()

    def save_group_report(self, output_dir):
        """Save a small JSON report with group counts, score means, and weights."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "method": "qmix_like",
            "group_counts": self.group_counts,
            "group_score_means": self.group_score_means,
            "weights": self.GROUP_WEIGHTS,
            "grouping_strategy": self.grouping_strategy,
        }
        with open(output_dir / "group_report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

    def _fit_gaussian_mixture(self, scores):
        if scores.numel() < self.num_groups or torch.unique(scores).numel() < self.num_groups:
            return None

        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            return None

        try:
            values = scores.numpy().reshape(-1, 1)
            model = GaussianMixture(
                n_components=self.num_groups,
                random_state=self.random_state,
            )
            labels = model.fit_predict(values)
        except Exception:
            return None

        component_means = []
        for component in range(self.num_groups):
            component_scores = scores[torch.as_tensor(labels == component)]
            if component_scores.numel() == 0:
                return None
            component_means.append((component, float(component_scores.mean().item())))

        component_means.sort(key=lambda item: item[1])
        component_to_group = {
            component: group for (component, _), group in zip(component_means, self.GROUP_ORDER)
        }
        return [component_to_group[int(label)] for label in labels]

    def _fit_percentiles(self, scores):
        order = torch.argsort(scores).tolist()
        groups = ["medium"] * scores.numel()
        total = max(scores.numel(), 1)

        for rank, score_index in enumerate(order):
            fraction = rank / total
            if fraction < 1.0 / 3.0:
                group = "clean"
            elif fraction < 2.0 / 3.0:
                group = "medium"
            else:
                group = "noisy"
            groups[score_index] = group

        return groups

    def _update_report(self, groups, scores):
        self.group_counts = {group: 0 for group in self.GROUP_ORDER}
        grouped_scores = {group: [] for group in self.GROUP_ORDER}

        for group, score in zip(groups, scores.tolist()):
            self.group_counts[group] += 1
            grouped_scores[group].append(float(score))

        self.group_score_means = {}
        for group in self.GROUP_ORDER:
            values = grouped_scores[group]
            if values:
                self.group_score_means[group] = float(sum(values) / len(values))
            else:
                self.group_score_means[group] = 0.0

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            if "index" not in batch:
                raise KeyError("QMix-like scoring requires batch['index'].")
            return batch["image"], batch["label"], batch["index"]

        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]

        raise ValueError("QMix-like scoring requires image, label, and index tensors.")
