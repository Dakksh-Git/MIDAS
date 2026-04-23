"""Training loop for the multimodal 3D brain MRI classifier."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "Data" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "Data" / "splits" / "val.csv"
SPLITS_DIR = PROJECT_ROOT / "Data" / "splits"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
NUM_CLASSES = 5
PATIENCE = 20
CLASS_NAMES = {
    0: "Malignant",
    1: "Benign",
    2: "Normal",
    3: "Scar",
    4: "Inflammatory",
}

CLASS_WEIGHTS = [
    1.5,   # Malignant
    0.8,   # Benign
    1.0,   # Normal
    0.50,  # Scar
    1.78,  # Inflammatory
]


def ensure_dirs() -> None:
    """Create required output directories if they do not exist."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


class BrainMRIDataset(Dataset):
    """Dataset for loading 4-channel 3D brain MRI volumes from CSV splits."""

    def __init__(self, csv_path: Path, augment: bool = False) -> None:
        """Load metadata from a CSV split.

        Args:
            csv_path: Path to the CSV containing filepath and label columns.
            augment: Whether to apply augmentation. Reserved for future use.
        """
        self.csv_path = Path(csv_path)
        self.augment = augment
        self.dataframe = pd.read_csv(self.csv_path)

        required_columns = {"filepath", "label"}
        missing_columns = required_columns - set(self.dataframe.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in {self.csv_path}: {sorted(missing_columns)}")

        self.filepaths = self.dataframe["filepath"].astype(str).tolist()
        self.labels = self.dataframe["label"].astype(int).tolist()

    def __len__(self) -> int:
        """Return the number of records in the split."""
        return len(self.dataframe)

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize a channel to [0, 1] using min-max scaling."""
        channel = channel.astype(np.float32, copy=False)
        min_value = float(channel.min())
        max_value = float(channel.max())
        if np.isclose(max_value, min_value):
            return np.zeros_like(channel, dtype=np.float32)
        normalized = (channel - min_value) / (max_value - min_value)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a sample tensor and its label.

        Args:
            idx: Dataset index.

        Returns:
            A tuple of (image_tensor, label_tensor). Missing files return a zero tensor and label -1.
        """
        filepath = Path(self.filepaths[idx])
        label = int(self.labels[idx])

        try:
            array = np.load(filepath)
            if array.shape != (4, 128, 128, 128):
                raise ValueError(f"Unexpected shape {array.shape} for file {filepath}")

            array = array.astype(np.float32, copy=False)
            normalized_channels = [self._normalize_channel(array[channel]) for channel in range(array.shape[0])]
            normalized = np.stack(normalized_channels, axis=0)
            tensor = torch.from_numpy(normalized)
            label_tensor = torch.tensor(label, dtype=torch.long)
            return tensor, label_tensor
        except FileNotFoundError:
            zero_tensor = torch.zeros((4, 128, 128, 128), dtype=torch.float32)
            return zero_tensor, torch.tensor(-1, dtype=torch.long)
        except Exception:
            zero_tensor = torch.zeros((4, 128, 128, 128), dtype=torch.float32)
            return zero_tensor, torch.tensor(-1, dtype=torch.long)


def build_class_weights(device: torch.device) -> torch.Tensor:
    """Build the manually specified normalized class-weight tensor."""
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
    weights = weights / weights.sum()
    return weights.to(device)


def get_device() -> torch.device:
    """Select the training device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    use_cuda = torch.cuda.is_available()
    pin_memory = bool(use_cuda)

    train_dataset = BrainMRIDataset(TRAIN_CSV, augment=False)
    val_dataset = BrainMRIDataset(VAL_CSV, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return train_loader, val_loader


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    val_acc: float,
    train_loss: float,
    class_weights: torch.Tensor,
) -> None:
    """Save a training checkpoint to disk."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "train_loss": train_loss,
        "class_weights": class_weights.detach().cpu(),
    }
    torch.save(checkpoint, path)


def compute_accuracy(predictions: list[int], targets: list[int]) -> float:
    """Compute accuracy from prediction and target label lists."""
    if not targets:
        return 0.0
    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    return correct / len(targets)


def compute_per_class_accuracy(predictions: list[int], targets: list[int]) -> dict[int, float]:
    """Compute per-class validation accuracy."""
    correct_by_class = defaultdict(int)
    total_by_class = defaultdict(int)

    for prediction, target in zip(predictions, targets):
        total_by_class[target] += 1
        if prediction == target:
            correct_by_class[target] += 1

    per_class_acc: dict[int, float] = {}
    for class_index in range(NUM_CLASSES):
        total = total_by_class.get(class_index, 0)
        per_class_acc[class_index] = 0.0 if total == 0 else correct_by_class.get(class_index, 0) / total
    return per_class_acc


def format_per_class_accuracy(per_class_acc: dict[int, float]) -> str:
    """Format per-class validation accuracy for logging."""
    lines = []
    for class_index in range(NUM_CLASSES):
        lines.append(f"  {CLASS_NAMES[class_index]}: {per_class_acc.get(class_index, 0.0) * 100:.2f}%")
    return "\n".join(lines)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in progress:
        valid_mask = labels != -1
        if not valid_mask.any():
            continue

        inputs = inputs[valid_mask]
        labels = labels[valid_mask]
        if inputs.size(0) < 2:
            continue

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return average_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[int, float]]:
    """Run one validation epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Validation", leave=False)
        for inputs, labels in progress:
            valid_mask = labels != -1
            if not valid_mask.any():
                continue

            inputs = inputs[valid_mask]
            labels = labels[valid_mask]
            if inputs.size(0) < 2:
                continue

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

            all_predictions.extend(predictions.detach().cpu().tolist())
            all_targets.extend(labels.detach().cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    per_class_accuracy = compute_per_class_accuracy(all_predictions, all_targets)
    return average_loss, accuracy, per_class_accuracy


def print_cuda_memory(device: torch.device) -> None:
    """Print CUDA memory usage if running on a CUDA device."""
    if not device.type == "cuda":
        return

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(
        f"CUDA memory | allocated: {allocated:.2f} MB | reserved: {reserved:.2f} MB | max allocated: {max_allocated:.2f} MB"
    )


def train_model(fresh: bool = False) -> dict[str, Any]:
    """Run the full training loop and return the training history."""
    ensure_dirs()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders()
    class_weights = build_class_weights(device)
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    model = get_model(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history_path = LOG_DIR / "training_history.json"
    checkpoint_path = CHECKPOINT_DIR / "last_model.pth"

    history: dict[str, list[Any]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "per_class_acc": [],
        "lr": [],
    }

    if not fresh and history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as handle:
                loaded_history = json.load(handle)
            if isinstance(loaded_history, dict):
                for key in history:
                    values = loaded_history.get(key, [])
                    history[key] = values if isinstance(values, list) else []
        except Exception:
            pass

    start_epoch = 1
    last_epoch_completed = len(history["train_loss"])
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    best_epoch = history["val_loss"].index(best_val_loss) + 1 if history["val_loss"] else -1

    if fresh:
        for path in [CHECKPOINT_DIR / "last_model.pth", CHECKPOINT_DIR / "best_model.pth"]:
            if path.exists():
                path.unlink()
        print("Fresh training requested; ignoring previous checkpoints and starting from epoch 1.")

    if not fresh and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1
        last_epoch_completed = max(last_epoch_completed, checkpoint_epoch)
        best_val_loss = min(best_val_loss, float(checkpoint.get("val_loss", best_val_loss)))

        print(f"Resuming training from checkpoint: {checkpoint_path}")
        print(f"Starting from epoch {start_epoch}/{NUM_EPOCHS}")

        if start_epoch > 1:
            for _ in range(start_epoch - 1):
                scheduler.step()

    if start_epoch > NUM_EPOCHS:
        print("Training is already complete based on the saved checkpoint.")
        return history

    patience_counter = 0
    start_time = time.time()

    try:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            epoch_start = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, per_class_acc = validate_one_epoch(model, val_loader, criterion, device)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["per_class_acc"].append({str(key): value for key, value in per_class_acc.items()})
            history["lr"].append(current_lr)

            print(f"[Epoch {epoch}/{NUM_EPOCHS}]")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc * 100:.2f}%")
            print("Per-class val accuracy:")
            print(format_per_class_accuracy(per_class_acc))
            print(f"LR: {current_lr:.8f}")
            print(f"Epoch time: {time.time() - epoch_start:.2f}s")
            print_cuda_memory(device)

            last_epoch_completed = epoch
            print(f"Epoch {epoch} complete")

            save_checkpoint(
                CHECKPOINT_DIR / "last_model.pth",
                epoch,
                model,
                optimizer,
                val_loss,
                val_acc,
                train_loss,
                class_weights,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    CHECKPOINT_DIR / "best_model.pth",
                    epoch,
                    model,
                    optimizer,
                    val_loss,
                    val_acc,
                    train_loss,
                    class_weights,
                )
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving last checkpoint before exit...")
        last_epoch = last_epoch_completed
        last_train_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
        last_val_loss = history["val_loss"][-1] if history["val_loss"] else 0.0
        last_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
        save_checkpoint(
            CHECKPOINT_DIR / "last_model.pth",
            last_epoch,
            model,
            optimizer,
            last_val_loss,
            last_val_acc,
            last_train_loss,
            class_weights,
        )
        raise
    finally:
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        total_time = time.time() - start_time
        print(f"Training history saved to {history_path}")
        print(f"Best epoch: {best_epoch}")
        print(f"Completed epochs: {last_epoch_completed}")
        print(f"Total training time: {total_time / 60:.2f} minutes")

    return history


if __name__ == "__main__":
    """Entry point for training."""
    parser = argparse.ArgumentParser(description="Train the multimodal 3D brain MRI classifier")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a fresh run from epoch 1 and skip loading checkpoints.",
    )
    args = parser.parse_args()
    train_model(fresh=args.fresh)
