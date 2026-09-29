"""Train-only synthetic augmentation for underrepresented classes.

Runs strictly AFTER src/rebuild_splits.py. Reads Data/splits/train.csv,
generates augmented derivatives only from volumes already assigned to the
train split, and appends them back into train.csv. val.csv and test.csv are
never read or written here, so no augmented volume can ever be a derivative
of a val/test original -- the leakage path that existed in the previous
augment.py pipeline (which pooled originals + augmented files and only THEN
split them into train/val/test) is structurally impossible with this design.

Augmentation recipe follows the paper's stated method (Section 3B):
  - random 3D rotation within +/-15 degrees
  - left-right reflection
  - additive Gaussian noise (sigma = 0.01)
  - multiplicative intensity perturbation within [0.9, 1.1]
Each generated volume is validated against the paper's four quality
criteria before being admitted.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "Data" / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
AUGMENTED_DIR = PROJECT_ROOT / "Data" / "processed" / "MRI" / "augmented_grouped"
SEED = 42

CLASS_NAMES = {
    0: "Malignant",
    1: "Benign",
    2: "Normal",
    3: "Scar",
    4: "Inflammatory",
}

# Same augmentation ratios the paper reports for the original (leaky) pool
# -- Benign 115->406 (~3.53x), Inflammatory 60->200 (~3.33x) -- applied to
# the new patient-grouped train-fold raw counts, so the class-imbalance
# mitigation intent is preserved without re-introducing leakage.
AUGMENT_RATIOS = {
    1: 406 / 115,
    4: 200 / 60,
}


def augment_rotation(volume: np.ndarray) -> np.ndarray:
    angle = random.uniform(-15, 15)
    rotated = np.zeros_like(volume)
    for ch in range(volume.shape[0]):
        rotated[ch] = ndimage.rotate(volume[ch], angle, reshape=False, order=1, mode="nearest")
    return np.clip(rotated, 0, 1)


def augment_flip(volume: np.ndarray) -> np.ndarray:
    return np.flip(volume, axis=1).copy()  # left-right reflection


def augment_intensity(volume: np.ndarray) -> np.ndarray:
    scaled = volume.copy()
    factor = random.uniform(0.9, 1.1)
    for ch in range(volume.shape[0]):
        scaled[ch] = volume[ch] * factor
    return np.clip(scaled, 0, 1)


def augment_gaussian_noise(volume: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0.0, 0.01, size=volume.shape).astype(np.float32)
    return np.clip(volume + noise, 0, 1)


def apply_augmentation_pipeline(volume: np.ndarray) -> np.ndarray:
    result = augment_rotation(volume)
    if random.random() < 0.5:
        result = augment_flip(result)
    result = augment_intensity(result)
    result = augment_gaussian_noise(result)
    return result.astype(np.float32)


def passes_quality_checks(volume: np.ndarray) -> bool:
    if volume.shape != (4, 128, 128, 128):
        return False
    if not np.isfinite(volume).all():
        return False
    for ch in range(4):
        std = float(np.std(volume[ch]))
        mean = float(np.mean(volume[ch]))
        if std <= 0.01:
            return False
        if not (0.05 < mean < 0.95):
            return False
    return True


def augment_class(train_df: pd.DataFrame, label: int, target_count: int) -> list[dict]:
    class_rows = train_df[(train_df["label"] == label) & (~train_df["is_augmented"])]
    source_paths = [Path(p) for p in class_rows["filepath"].tolist()]
    current_count = len(source_paths)
    needed = max(0, target_count - current_count)

    print(f"\nClass {label} ({CLASS_NAMES[label]}): raw train={current_count}, "
          f"target={target_count}, generating={needed}")

    if needed == 0 or not source_paths:
        return []

    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    new_records: list[dict] = []
    aug_index = 0
    generated = 0

    with tqdm(total=needed, desc=f"Augmenting class {label}", unit="sample") as pbar:
        attempts = 0
        max_attempts = needed * 20 + 50
        while generated < needed and attempts < max_attempts:
            attempts += 1
            original_path = random.choice(source_paths)
            try:
                volume = np.load(original_path).astype(np.float32)
            except Exception as exc:
                print(f"Failed to load {original_path}: {exc}")
                continue

            if volume.shape != (4, 128, 128, 128):
                continue

            aug_volume = apply_augmentation_pipeline(volume)
            if not passes_quality_checks(aug_volume):
                continue

            out_name = f"aug_{original_path.stem}_{aug_index}.npy"
            out_path = AUGMENTED_DIR / out_name
            np.save(out_path, aug_volume)

            new_records.append({
                "filepath": str(out_path.resolve()),
                "label": label,
                "source": "augmented_grouped",
                "is_augmented": True,
            })

            aug_index += 1
            generated += 1
            pbar.update(1)

        if attempts >= max_attempts and generated < needed:
            print(f"WARNING: stopped after {attempts} attempts, only generated "
                  f"{generated}/{needed} for class {label}.")

    print(f"  Generated {generated} augmented volumes for class {label} ({CLASS_NAMES[label]})")
    return new_records


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    if not TRAIN_CSV.exists():
        raise SystemExit(f"{TRAIN_CSV} not found. Run src/rebuild_splits.py first.")

    train_df = pd.read_csv(TRAIN_CSV)
    if "is_augmented" not in train_df.columns:
        train_df["is_augmented"] = False

    all_new_records: list[dict] = []
    for label, ratio in AUGMENT_RATIOS.items():
        raw_count = len(train_df[(train_df["label"] == label) & (~train_df["is_augmented"])])
        target_count = round(raw_count * ratio)
        all_new_records.extend(augment_class(train_df, label, target_count))

    if all_new_records:
        updated_df = pd.concat([train_df, pd.DataFrame(all_new_records)], ignore_index=True)
    else:
        updated_df = train_df

    updated_df.to_csv(TRAIN_CSV, index=False)

    print("\nFinal train split composition:")
    counts = Counter(updated_df["label"])
    aug_counts = Counter(updated_df[updated_df["is_augmented"]]["label"])
    for label in sorted(CLASS_NAMES):
        total = counts.get(label, 0)
        aug = aug_counts.get(label, 0)
        print(f"  Class {label} ({CLASS_NAMES[label]}): {total} total ({aug} augmented)")
    print(f"\nTotal train volumes: {len(updated_df)}")
    print(f"Updated: {TRAIN_CSV}")


if __name__ == "__main__":
    main()
