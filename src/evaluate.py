from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = PROJECT_ROOT / "Data" / "splits" / "test.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLASS_NAMES = {0: "Malignant", 1: "Benign", 2: "Normal", 3: "Scar", 4: "Inflammatory"}
NUM_CLASSES = 5
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BrainMRIDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.dataframe = pd.read_csv(csv_path)
        required_columns = {"filepath", "label"}
        missing = required_columns - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        self.filepaths = self.dataframe["filepath"].astype(str).tolist()
        self.labels = self.dataframe["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        array = np.load(self.filepaths[idx]).astype(np.float32)
        tensor = torch.from_numpy(array)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


def format_confusion_matrix(cm: np.ndarray) -> str:
    headers = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    width = max(max(len(h) for h in headers), 12)

    lines = []
    header_row = " " * (width + 2) + " ".join(f"{h:>{width}}" for h in headers)
    lines.append(header_row)

    for i in range(NUM_CLASSES):
        row_name = CLASS_NAMES[i]
        row_values = " ".join(f"{int(cm[i, j]):>{width}}" for j in range(NUM_CLASSES))
        lines.append(f"{row_name:>{width}}: {row_values}")

    return "\n".join(lines)


def evaluate() -> None:
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = BrainMRIDataset(TEST_CSV)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_model(device=DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", unit="batch"):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, NUM_CLASSES), dtype=np.float32)

    overall_acc = accuracy_score(y_true, y_pred)
    per_class_acc = {}
    for cls in range(NUM_CLASSES):
        cls_mask = y_true == cls
        if int(cls_mask.sum()) == 0:
            per_class_acc[cls] = 0.0
        else:
            per_class_acc[cls] = float((y_pred[cls_mask] == y_true[cls_mask]).mean())

    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    auc_macro_ovr = roc_auc_score(
        y_true,
        y_prob,
        labels=list(range(NUM_CLASSES)),
        multi_class="ovr",
        average="macro",
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    print("\n" + "=" * 80)
    print("TEST EVALUATION RESULTS")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Test samples: {len(dataset)}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro AUC-ROC (OvR): {auc_macro_ovr:.4f}")

    print("\nPer-class Accuracy:")
    for cls in range(NUM_CLASSES):
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {per_class_acc[cls]:.4f}")

    print("\nPer-class F1:")
    for cls in range(NUM_CLASSES):
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {float(per_class_f1[cls]):.4f}")

    print("\nConfusion Matrix:")
    print(format_confusion_matrix(cm))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        digits=4,
        zero_division=0,
    )

    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        yticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print("\nSaved files:")
    print(f"  - {cm_path}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    evaluate()
