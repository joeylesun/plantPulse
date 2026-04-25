import numpy as np
import torch
from torch.utils.data import DataLoader


class RandomBaseline:
    """Predicts a uniformly random class for each input.

    Also supports a 'stratified' mode that samples classes in proportion to
    their frequency in the training set - useful if classes are imbalanced.
    """

    def __init__(self, num_classes: int, mode: str = "uniform", class_weights=None, seed: int = 42):
        self.num_classes = num_classes
        self.mode = mode
        self.class_weights = class_weights
        self.rng = np.random.default_rng(seed)

    def predict(self, n: int) -> np.ndarray:
        """Generate n random predictions."""
        if self.mode == "uniform":
            return self.rng.integers(0, self.num_classes, size=n)
        elif self.mode == "stratified":
            if self.class_weights is None:
                raise ValueError("class_weights required for stratified mode")
            return self.rng.choice(self.num_classes, size=n, p=self.class_weights)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def evaluate_random_baseline(loader: DataLoader, num_classes: int) -> float:
    """Compute accuracy of a uniform random baseline on the given loader."""
    baseline = RandomBaseline(num_classes=num_classes, mode="uniform")
    total = 0
    correct = 0
    for _, labels in loader:
        preds = baseline.predict(len(labels))
        correct += (torch.tensor(preds) == labels).sum().item()
        total += len(labels)
    return correct / total
