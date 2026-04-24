"""
Training loop with train/val loss and accuracy tracking (for learning curves).

Rubric items addressed here:
  - Training curves (returns history dict with per-epoch losses + accuracies)
  - Best-model checkpointing based on validation accuracy

AI attribution: Scaffolding drafted with Claude (Anthropic), reviewed by author.
"""

from pathlib import Path
from typing import Dict, List, Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    train: bool,
):
    """Run one epoch. If train=True we update weights; otherwise we just eval."""
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # torch.set_grad_enabled lets us share this function for train and eval
    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """Fine-tune the model, tracking metrics each epoch.

    Returns a history dict containing per-epoch train/val loss and accuracy,
    which the notebooks use to plot learning curves.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that require grad. This matters for the frozen-
    # backbone ablation - the backbone params have requires_grad=False there,
    # so we only update the head.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save the best checkpoint so far (measured by validation accuracy)
        if save_path and val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "class_names": class_names},
                save_path,
            )
            print(f"  -> new best; saved to {save_path}")

    return history
