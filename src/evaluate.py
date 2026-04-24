"""
Evaluation utilities: test-set accuracy, per-class accuracy, confusion matrix.

These are used by notebooks 03 (ablation) and 04 (error analysis).
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model on a loader and return (y_true, y_pred, probs).

    probs is the softmax confidence per sample, useful for error analysis.
    """
    model.eval()
    model.to(device)

    all_true, all_pred, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_true.append(labels.numpy())
        all_pred.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Overall accuracy, per-class accuracy, confusion matrix, classification report."""
    overall_acc = (y_true == y_pred).mean()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Per-class accuracy = diagonal / row sum (recall per class)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.where(cm.sum(axis=1) > 0, np.diag(cm) / cm.sum(axis=1), 0)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": dict(zip(class_names, per_class_acc.tolist())),
        "confusion_matrix": cm,
        "classification_report": report,
    }
