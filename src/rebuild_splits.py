"""Rebuild train/val/test splits at the PATIENT level.

Replaces the earlier file-level stratified split (preserved as
rebuild_splits_filelevel_OLD.py), which stratified individual .npy files
without regard to which patient/subject they came from. That approach let
the same patient's scans (e.g. multiple LUMIERE timepoints) land in both
train and test, and let augmented derivatives of a training volume land in
val/test (or vice versa) -- both are data leakage.

This script:
  1. Scans only RAW (non-augmented) processed volumes.
  2. Extracts a patient/subject key per source (regexes verified against the
     actual filenames produced by src/preprocess.py).
  3. Splits at the GROUP (patient) level, stratified by class, so every file
     belonging to one patient stays in exactly one of train/val/test.
  4. Writes Data/splits/{train,val,test}.csv containing RAW volumes only.
     Augmentation is handled separately by src/augment_train.py, which only
     ever reads train.csv and only ever writes back into train.csv.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed" / "MRI"
SPLITS_DIR = PROJECT_ROOT / "Data" / "splits"
RANDOM_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is the remainder (0.15)

SOURCES = {
    "brats": PROCESSED_DIR / "brats",
    "remind": PROCESSED_DIR / "remind",
    "ixi": PROCESSED_DIR / "ixi",
    "lumiere": PROCESSED_DIR / "lumiere",
    "ms": PROCESSED_DIR / "ms",
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


def extract_label(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return None


def patient_key(filename: str, source_name: str) -> Optional[str]:
    """Extract a patient/subject grouping key so no single patient's files
    can be split across train/val/test."""
    stem = Path(filename).stem

    if source_name == "brats":
        m = re.search(r"BraTS20_Training_(\d+)", stem)
        return f"brats_{m.group(1)}" if m else None

    if source_name == "remind":
        m = re.search(r"ReMIND-(\d+)", stem)
        return f"remind_{m.group(1)}" if m else None

    if source_name == "ixi":
        m = re.search(r"IXI(\d+)", stem)
        return f"ixi_{m.group(1)}" if m else None

    if source_name == "lumiere":
        # lumiere_Patient-XXX_week-YYY(-Z)?_3.npy -- group by patient only,
        # NOT by week, so all longitudinal timepoints of one patient stay together.
        m = re.search(r"Patient-(\d+)", stem)
        return f"lumiere_{m.group(1)}" if m else None

    if source_name == "ms":
        m = re.search(r"^ms_(\d+)_\d+$", stem)
        return f"ms_{m.group(1)}" if m else None

    return None


def build_records() -> List[dict]:
    records: List[dict] = []

    for source_name, source_dir in SOURCES.items():
        if not source_dir.exists() or not source_dir.is_dir():
            continue

        for file_path in sorted(source_dir.glob("*.npy")):
            filename = file_path.name

            # Match the original pipeline: BraTS "normal" 2D slices were never
            # part of the modelled Normal class (Normal comes entirely from IXI).
            if source_name == "brats" and "_normal_" in filename:
                continue

            label = extract_label(filename)
            if label is None:
                continue

            key = patient_key(filename, source_name)
            if key is None:
                print(f"WARNING: could not extract patient key for {filename}; skipping.")
                continue

            records.append(
                {
                    "filepath": str(file_path.resolve()),
                    "label": int(label),
                    "source": source_name,
                    "is_augmented": False,
                    "patient_key": key,
                }
            )

    return records


def group_stratified_split(records: List[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split patient GROUPS (not files) into train/val/test, stratified by class.

    Each patient key is assigned to exactly one split. Every file under that
    key follows its group, so no patient's data crosses partitions.
    """
    if not records:
        raise ValueError("No records found to split.")

    # Map each patient_key -> its class label (patients are single-label in
    # this dataset; verified against src/preprocess.py's labelling logic).
    key_to_label: Dict[str, int] = {}
    key_to_records: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        key = rec["patient_key"]
        key_to_records[key].append(rec)
        if key in key_to_label and key_to_label[key] != rec["label"]:
            raise ValueError(
                f"Patient key {key} maps to multiple labels "
                f"({key_to_label[key]} and {rec['label']}) -- grouping assumption violated."
            )
        key_to_label[key] = rec["label"]

    label_to_keys: Dict[int, List[str]] = defaultdict(list)
    for key, label in key_to_label.items():
        label_to_keys[label].append(key)

    rng = random.Random(RANDOM_SEED)
    train_records: List[dict] = []
    val_records: List[dict] = []
    test_records: List[dict] = []

    for label in sorted(label_to_keys):
        keys = sorted(label_to_keys[label])  # deterministic order before shuffle
        rng.shuffle(keys)

        n = len(keys)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        # Guard against rounding pushing val/test allocation negative or over n.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train_keys = set(keys[:n_train])
        val_keys = set(keys[n_train:n_train + n_val])
        test_keys = set(keys[n_train + n_val:])

        for key in train_keys:
            train_records.extend(key_to_records[key])
        for key in val_keys:
            val_records.extend(key_to_records[key])
        for key in test_keys:
            test_records.extend(key_to_records[key])

    return train_records, val_records, test_records


def to_rows(records: List[dict]) -> List[dict]:
    return [
        {
            "filepath": r["filepath"],
            "label": r["label"],
            "source": r["source"],
            "is_augmented": r["is_augmented"],
        }
        for r in records
    ]


def print_split_summary(name: str, records: List[dict]) -> None:
    counts = Counter(r["label"] for r in records)
    n_patients = len({r["patient_key"] for r in records})
    print(f"{name}: {len(records)} volumes across {n_patients} patients")
    for label in sorted(CLASS_NAMES):
        print(f"  Class {label} ({CLASS_NAMES[label]}): {counts.get(label, 0)}")


def save_splits(train_records: List[dict], val_records: List[dict], test_records: List[dict]) -> None:
    import pandas as pd

    ensure_dir(SPLITS_DIR)
    pd.DataFrame(to_rows(train_records)).to_csv(SPLITS_DIR / "train.csv", index=False)
    pd.DataFrame(to_rows(val_records)).to_csv(SPLITS_DIR / "val.csv", index=False)
    pd.DataFrame(to_rows(test_records)).to_csv(SPLITS_DIR / "test.csv", index=False)


def verify_no_leakage(train_records: List[dict], val_records: List[dict], test_records: List[dict]) -> None:
    train_keys = {r["patient_key"] for r in train_records}
    val_keys = {r["patient_key"] for r in val_records}
    test_keys = {r["patient_key"] for r in test_records}

    overlaps = {
        "train<->val": train_keys & val_keys,
        "train<->test": train_keys & test_keys,
        "val<->test": val_keys & test_keys,
    }
    for pair, overlap in overlaps.items():
        if overlap:
            raise AssertionError(f"Patient leakage detected {pair}: {sorted(overlap)[:10]}")
    print("\nVerified: zero patient overlap between train/val/test.")


def main() -> None:
    records = build_records()
    if not records:
        raise SystemExit("No records found under Data/processed/MRI.")

    train_records, val_records, test_records = group_stratified_split(records)
    verify_no_leakage(train_records, val_records, test_records)
    save_splits(train_records, val_records, test_records)

    print_split_summary("Train", train_records)
    print_split_summary("Val", val_records)
    print_split_summary("Test", test_records)
    print(f"\nSaved patient-grouped splits to: {SPLITS_DIR}")
    print("Note: these splits contain RAW volumes only. Run src/augment_train.py next")
    print("to add train-only synthetic augmentation for underrepresented classes.")


if __name__ == "__main__":
    main()
