"""
Evaluation metrics for geohazard detection.

Implements:
  - Standard classification metrics (accuracy, precision, recall, F1)
  - Per-class metrics (important for imbalanced hazard datasets)
  - Confusion matrix generation
  - ROC-AUC per class
  - Macro / Micro / Weighted averaging
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsTracker:
    """Tracks and reports classification metrics during training/evaluation."""

    def __init__(self, class_names: list[str]):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.all_preds: list[int] = []
        self.all_labels: list[int] = []
        self.all_probs: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """Accumulate batch predictions."""
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        self.all_preds.extend(preds.cpu().tolist())
        self.all_labels.extend(labels.cpu().tolist())
        self.all_probs.extend(probs.cpu().numpy())

    def compute(self) -> dict:
        """Compute all metrics from accumulated predictions."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        y_prob = np.stack(self.all_probs, axis=0)

        n_classes = len(self.class_names)

        # Per-class and aggregated metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

        # ROC-AUC (one-vs-rest) — handles missing classes gracefully
        try:
            if y_prob.shape[1] == n_classes:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
        except ValueError:
            metrics["roc_auc"] = float("nan")

        # Per-class breakdown
        per_class = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0,
            output_dict=True,
        )
        metrics["per_class"] = per_class

        return metrics

    def log(self, prefix: str = ""):
        """Compute and log all metrics."""
        m = self.compute()
        logger.info(f"{prefix} Accuracy: {m['accuracy']:.4f}")
        logger.info(f"{prefix} F1 (macro): {m['f1_macro']:.4f}")
        logger.info(f"{prefix} F1 (weighted): {m['f1_weighted']:.4f}")
        if not np.isnan(m.get("roc_auc", float("nan"))):
            logger.info(f"{prefix} ROC-AUC: {m['roc_auc']:.4f}")
        return m

    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot and optionally save confusion matrix."""
        m = self.compute()
        cm = m["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — GeoHazard Detection")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.close(fig)
