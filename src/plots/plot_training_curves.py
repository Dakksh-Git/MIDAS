from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "outputs" / "logs" / "training_history.json"
OUTPUT_PATH = PROJECT_ROOT / "fig4_1_training_curves.png"
BEST_EPOCH = 27


def _to_percent_if_needed(values: list[float]) -> list[float]:
    if not values:
        return values
    max_value = max(values)
    if max_value <= 1.0:
        return [v * 100.0 for v in values]
    return values


def load_history() -> tuple[list[float], list[float], list[float], list[float]]:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Training history file not found: {HISTORY_PATH}")

    payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training_history.json must contain a JSON object.")

    required_keys = ["train_loss", "train_acc", "val_loss", "val_acc", "per_class_acc", "lr"]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise KeyError(f"Missing required keys in training_history.json: {missing}")

    try:
        train_loss = [float(v) for v in payload["train_loss"]]
        val_loss = [float(v) for v in payload["val_loss"]]
        train_acc = [float(v) for v in payload["train_acc"]]
        val_acc = [float(v) for v in payload["val_acc"]]
    except (TypeError, ValueError) as exc:
        raise ValueError("train_loss, val_loss, train_acc, and val_acc must be numeric lists.") from exc

    if not (len(train_loss) == len(val_loss) == len(train_acc) == len(val_acc)):
        raise ValueError("train_loss, val_loss, train_acc, and val_acc must have the same length.")

    return train_loss, val_loss, train_acc, val_acc


def plot_training_curves() -> Path:
    train_loss, val_loss, train_acc, val_acc = load_history()
    train_acc = _to_percent_if_needed(train_acc)
    val_acc = _to_percent_if_needed(val_acc)

    epochs_loss = np.arange(1, len(train_loss) + 1)
    epochs_acc = np.arange(1, len(train_acc) + 1)

    plt.rcParams["font.family"] = "Times New Roman"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.patch.set_facecolor("white")

    ax1.plot(epochs_loss, train_loss, color="tab:blue", linewidth=2.0, label="Training Loss")
    ax1.plot(epochs_loss, val_loss, color="tab:orange", linewidth=2.0, linestyle="--", label="Validation Loss")
    ax1.axvline(BEST_EPOCH, color="red", linestyle="--", linewidth=1.5)
    loss_y_max = max(max(train_loss), max(val_loss))
    ax1.text(BEST_EPOCH + 0.4, loss_y_max * 0.95, "Best Model (Epoch 27)", color="red", fontsize=10)
    ax1.set_xlim(1, len(train_loss))
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.grid(True, color="lightgrey", alpha=0.4)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.legend(loc="lower right", fontsize=12)

    ax2.plot(epochs_acc, train_acc, color="tab:blue", linewidth=2.0, label="Training Accuracy")
    ax2.plot(epochs_acc, val_acc, color="tab:orange", linewidth=2.0, linestyle="--", label="Validation Accuracy")
    ax2.axvline(BEST_EPOCH, color="red", linestyle="--", linewidth=1.5)
    acc_y_max = max(max(train_acc), max(val_acc))
    ax2.text(BEST_EPOCH + 0.4, min(104.0, acc_y_max + 1.0), "Best Model (Epoch 27)", color="red", fontsize=10)
    ax2.axhline(92.11, color="green", linestyle="--", linewidth=1.5, label="Test Accuracy (92.11%)")
    ax2.text(2, 92.11 + 0.6, "Test Accuracy (92.11%)", color="green", fontsize=10)
    ax2.set_xlim(1, len(train_acc))
    ax2.set_ylim(50, 105)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.grid(True, color="lightgrey", alpha=0.4)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.legend(loc="lower right", fontsize=12)

    fig.suptitle(
        "Fig. 4.1: Training and Validation Loss/Accuracy Curves — v5 Model (47 epochs, Best at Epoch 27)",
        fontsize=12,
        fontname="Times New Roman",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    saved = plot_training_curves()
    print(f"Saved figure to: {saved}")