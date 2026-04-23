from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed" / "MRI"
SPLITS_DIR = PROJECT_ROOT / "Data" / "splits"
RANDOM_SEED = 42

SOURCES = {
    "brats": PROCESSED_DIR / "brats",
    "remind": PROCESSED_DIR / "remind",
    "ixi": PROCESSED_DIR / "ixi",
    "lumiere": PROCESSED_DIR / "lumiere",
    "ms": PROCESSED_DIR / "ms",
    "augmented": PROCESSED_DIR / "augmented",
}

CLASS_NAMES = {
    0: "Malignant",
    1: "Benign",
    2: "Normal",
    3: "Scar",
    4: "Inflammatory",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_label(filename: str, source_name: str) -> int | None:
    stem = Path(filename).stem
    parts = stem.split("_")

    try:
        if source_name == "augmented":
            if len(parts) < 2:
                return None
            return int(parts[-2])
        return int(parts[-1])
    except (ValueError, IndexError):
        return None


def build_records() -> List[dict]:
    records: List[dict] = []

    for source_name, source_dir in SOURCES.items():
        if not source_dir.exists() or not source_dir.is_dir():
            continue

        for file_path in sorted(source_dir.rglob("*.npy")):
            filename = file_path.name
            label = extract_label(filename, source_name)
            if label is None:
                continue
            if source_name == "brats" and (label == 2 or "_normal_" in filename):
                continue

            records.append(
                {
                    "filepath": str(file_path.resolve()),
                    "label": int(label),
                    "source": source_name,
                    "is_augmented": source_name == "augmented",
                }
            )

    return records


def stratified_split(records: List[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not records:
        raise ValueError("No valid .npy files were found to build splits.")

    df = pd.DataFrame(records)[["filepath", "label", "source", "is_augmented"]]
    labels = df["label"]

    label_counts = Counter(labels)
    too_small = [label for label, count in label_counts.items() if count < 2]
    if too_small:
        raise ValueError(
            "Stratified split requires at least 2 samples per class; missing minimum count for labels: "
            f"{sorted(int(label) for label in too_small)}"
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=labels,
    )

    temp_labels = temp_df["label"]
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=temp_labels,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    print(f"{name}: {len(df)} samples")
    counts = df["label"].value_counts().sort_index()
    for label in sorted(CLASS_NAMES):
        print(f"  Class {label} ({CLASS_NAMES[label]}): {int(counts.get(label, 0))}")


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    ensure_dir(SPLITS_DIR)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)


def main() -> None:
    random.seed(RANDOM_SEED)

    records = build_records()
    if not records:
        raise SystemExit("No records found under Data/processed/MRI.")

    train_df, val_df, test_df = stratified_split(records)
    save_splits(train_df, val_df, test_df)

    print_split_summary("Train", train_df)
    print_split_summary("Val", val_df)
    print_split_summary("Test", test_df)
    print(f"\nSaved splits to: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
