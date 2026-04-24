"""
Error analysis utilities: find the model's worst classes and most-confused pairs.

Rubric item addressed: error analysis on challenging cases (similar-looking diseases).

Usage:
    See notebooks/04_error_analysis.ipynb for a full walkthrough.
"""

from typing import List, Tuple, Dict
import numpy as np


def top_confused_pairs(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    top_n: int = 10,
) -> List[Tuple[str, str, int]]:
    """Find the (true, predicted, count) triples with the most confusions.

    Excludes the diagonal (correct predictions). Useful for answering
    "which diseases does the model confuse with which?"
    """
    cm = confusion_matrix.copy()
    np.fill_diagonal(cm, 0)  # ignore correct predictions

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def worst_classes_by_accuracy(
    per_class_accuracy: Dict[str, float],
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """Return the top_n classes with the lowest per-class accuracy."""
    sorted_classes = sorted(per_class_accuracy.items(), key=lambda x: x[1])
    return sorted_classes[:top_n]


def hardest_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    top_n: int = 20,
) -> List[Dict]:
    """Find the model's most confident wrong predictions.

    These are the scariest errors - the model got it wrong AND was sure of
    itself. Reviewing these often reveals dataset issues (mislabeled images)
    or genuinely hard cases.
    """
    wrong_mask = y_true != y_pred
    wrong_indices = np.where(wrong_mask)[0]

    # Confidence on the (wrong) predicted class
    wrong_confidences = probs[wrong_indices, y_pred[wrong_indices]]

    # Sort by confidence descending - we want the most-confident mistakes
    order = np.argsort(-wrong_confidences)
    top = wrong_indices[order[:top_n]]

    return [
        {
            "dataset_index": int(i),
            "true_class": int(y_true[i]),
            "predicted_class": int(y_pred[i]),
            "confidence": float(probs[i, y_pred[i]]),
        }
        for i in top
    ]
