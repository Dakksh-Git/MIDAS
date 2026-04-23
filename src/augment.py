"""Data augmentation for underrepresented brain MRI classes.

This script performs augmentation on Benign (class 1), Scar (class 3), and
Inflammatory (class 4) samples, undersamples the Normal (class 2) class, and
regenerates train/val/test splits.
"""

from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed" / "MRI"
SPLITS_DIR = PROJECT_ROOT / "Data" / "splits"
AUGMENTED_DIR = PROCESSED_DIR / "augmented"
NORMAL_SELECTED_FILE = PROCESSED_DIR / "normal_selected.txt"

CLASS_NAMES = {
    0: "Malignant",
    1: "Benign",
    2: "Normal",
    3: "Scar",
    4: "Inflammatory",
}

AUGMENT_TARGETS = {
    1: 361,   # Benign -> augment up to 361
    3: 200,   # Scar -> already done, skip if exists
    4: 200,   # Inflammatory -> already done, skip if exists
}
AUGMENT_CLASSES = set(AUGMENT_TARGETS)
UNDERSAMPLE_CLASS = 2
UNDERSAMPLE_TARGET = 361


def collect_processed_files(patterns: list[str]) -> list[Path]:
    """Collect matching processed files outside the augmented directory."""
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(
            p for p in PROCESSED_DIR.rglob(pattern)
            if p.parent.name != "augmented" and not p.stem.startswith("aug_")
        )
    return sorted(set(matches), key=lambda p: str(p))


def summarize_existing_augmented_samples() -> tuple[Counter, dict[int, int]]:
    """Count existing augmented samples and track the next file index per class."""
    augmented_counts = Counter()
    next_aug_index: dict[int, int] = {}

    if not AUGMENTED_DIR.exists():
        return augmented_counts, next_aug_index

    max_indexes: dict[int, int] = defaultdict(lambda: -1)
    for aug_file in AUGMENTED_DIR.glob("aug_*.npy"):
        parts = aug_file.stem.split("_")
        if len(parts) < 3:
            continue

        try:
            label = int(parts[-2])
            index = int(parts[-1])
        except ValueError:
            continue

        if label not in CLASS_NAMES:
            continue

        augmented_counts[label] += 1
        max_indexes[label] = max(max_indexes[label], index)

    for label, max_index in max_indexes.items():
        next_aug_index[label] = max_index + 1

    return augmented_counts, next_aug_index


def count_total_existing_for_class(class_label: int, original_count: int) -> int:
    """Count total existing samples (original + already-augmented) for one class."""
    total = original_count

    # Primary augmented directory accounting.
    if AUGMENTED_DIR.exists():
        for aug_file in AUGMENTED_DIR.glob("aug_*.npy"):
            parts = aug_file.stem.split("_")
            if len(parts) < 3:
                continue
            try:
                label = int(parts[-2])
            except ValueError:
                continue
            if label == class_label:
                total += 1

    # Safety net: catch historical benign files saved outside augmented/.
    if class_label == 1:
        outside_aug = {
            p for p in PROCESSED_DIR.rglob("aug_benign_*.npy")
            if p.parent.name != "augmented"
        }
        total += len(outside_aug)

    return total


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def find_existing_samples() -> tuple[dict[int, list[Path]], Counter]:
    """Scan for existing .npy files and organize by class label."""
    class_samples: dict[int, list[Path]] = defaultdict(list)
    class_counts = Counter()

    print("\nScanning for existing samples...")
    for npy_file in PROCESSED_DIR.rglob("*.npy"):
        if npy_file.parent.name == "augmented":
            continue

        if npy_file.stem.startswith("aug_"):
            continue

        try:
            parts = npy_file.stem.split("_")
            if not parts:
                continue

            # Handle BraTS normal naming separately via explicit patterns below.
            if "_normal_" in npy_file.stem:
                continue

            label = int(parts[-1])
            if label in CLASS_NAMES:
                class_samples[label].append(npy_file)
                class_counts[label] += 1
        except (ValueError, IndexError):
            continue

    # Explicit Benign discovery to catch both BraTS and ReMIND conventions.
    benign_brats_pattern = "brats_*_1.npy"
    benign_remind_pattern = "remind_*_1.npy"
    print(f"\nBenign glob pattern (BraTS): {benign_brats_pattern}")
    print(f"Benign glob pattern (ReMIND): {benign_remind_pattern}")

    benign_files = collect_processed_files([benign_brats_pattern, benign_remind_pattern])
    class_samples[1] = benign_files
    class_counts[1] = len(benign_files)

    # Explicit normal discovery to catch both BraTS and ReMIND conventions.
    normal_brats_pattern = "*_normal_*.npy"
    normal_remind_pattern = "remind_*_2.npy"
    print(f"\nNormal glob pattern (BraTS): {normal_brats_pattern}")
    print(f"Normal glob pattern (ReMIND): {normal_remind_pattern}")

    normal_files = collect_processed_files([normal_brats_pattern, normal_remind_pattern])

    class_samples[2] = normal_files
    class_counts[2] = len(normal_files)

    print("\nFirst 5 Normal filepaths found:")
    if normal_files:
        for normal_path in normal_files[:5]:
            print(f"  - {normal_path}")
    else:
        print("  - None")
    print(f"Total Normal files found before undersampling: {len(normal_files)}")

    # Debug and deduplicate malignant paths.
    malignant_raw = class_samples.get(0, [])
    print("\nFirst 10 Malignant filepaths found (raw):")
    if malignant_raw:
        for malignant_path in malignant_raw[:10]:
            print(f"  - {malignant_path}")
    else:
        print("  - None")

    malignant_deduped = sorted(set(malignant_raw), key=lambda p: str(p))
    if len(malignant_deduped) != len(malignant_raw):
        print(
            f"Deduplicated Malignant filepaths: {len(malignant_raw)} -> {len(malignant_deduped)}"
        )
    class_samples[0] = malignant_deduped
    class_counts[0] = len(malignant_deduped)

    print("\nFirst 5 Benign filepaths found:")
    if benign_files:
        for benign_path in benign_files[:5]:
            print(f"  - {benign_path}")
    else:
        print("  - None")
    print(f"Total Benign files found before augmentation: {len(benign_files)}")

    print("\nExisting samples per class:")
    for label in sorted(CLASS_NAMES):
        count = class_counts.get(label, 0)
        print(f"  Class {label} ({CLASS_NAMES[label]}): {count}")

    return dict(class_samples), class_counts


def load_sample(filepath: Path) -> np.ndarray | None:
    """Load a .npy file safely."""
    try:
        array = np.load(filepath)
        if array.ndim != 4 or array.shape[0] != 4:
            return None
        return array.astype(np.float32)
    except Exception as exc:
        print(f"Error loading {filepath}: {exc}")
        return None


def augment_rotation(volume: np.ndarray) -> np.ndarray:
    """Apply random rotation to all 4 channels."""
    try:
        angle = random.uniform(-15, 15)
        rotated = np.zeros_like(volume)
        for ch in range(4):
            rotated[ch] = ndimage.rotate(volume[ch], angle, reshape=False, order=1)
        return np.clip(rotated, 0, 1)
    except Exception as exc:
        print(f"Rotation error: {exc}")
        return volume


def augment_flip(volume: np.ndarray) -> np.ndarray:
    """Apply random flip along a random axis."""
    try:
        axis = random.choice([1, 2, 3])
        flipped = np.flip(volume, axis=axis)
        return flipped.copy()
    except Exception as exc:
        print(f"Flip error: {exc}")
        return volume


def augment_intensity(volume: np.ndarray) -> np.ndarray:
    """Apply random intensity scaling to each channel."""
    try:
        scaled = volume.copy()
        for ch in range(4):
            factor = random.uniform(0.85, 1.15)
            scaled[ch] = volume[ch] * factor
        return np.clip(scaled, 0, 1)
    except Exception as exc:
        print(f"Intensity error: {exc}")
        return volume


def augment_gaussian_noise(volume: np.ndarray) -> np.ndarray:
    """Add random Gaussian noise to each channel."""
    try:
        noisy = volume.copy()
        for ch in range(4):
            std = random.uniform(0.01, 0.03)
            noise = np.random.normal(0, std, volume[ch].shape)
            noisy[ch] = volume[ch] + noise
        return np.clip(noisy, 0, 1)
    except Exception as exc:
        print(f"Gaussian noise error: {exc}")
        return volume


def augment_elastic_deformation(volume: np.ndarray) -> np.ndarray:
    """Apply elastic deformation to all 4 channels."""
    try:
        shape = volume[0].shape
        dx = np.random.randn(*shape) * 2
        dy = np.random.randn(*shape) * 2
        dz = np.random.randn(*shape) * 2

        x, y, z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )
        indices = np.array([x + dx, y + dy, z + dz])

        deformed = np.zeros_like(volume)
        for ch in range(4):
            deformed[ch] = ndimage.map_coordinates(
                volume[ch], indices, order=1, cval=0.0
            )
        return np.clip(deformed, 0, 1)
    except Exception as exc:
        print(f"Elastic deformation error: {exc}")
        return volume


def apply_random_augmentations(volume: np.ndarray) -> np.ndarray:
    """Apply 2-3 random augmentations to a volume."""
    augmentations = [
        augment_rotation,
        augment_flip,
        augment_intensity,
        augment_gaussian_noise,
        augment_elastic_deformation,
    ]

    num_augs = random.randint(2, 3)
    selected_augs = random.sample(augmentations, num_augs)

    result = volume.copy()
    for aug_func in selected_augs:
        result = aug_func(result)

    return result


def augment_class_samples(
    class_label: int,
    filepaths: list[Path],
    existing_total: int,
    next_aug_index: int,
) -> int:
    """Augment samples for a given class until reaching its target."""
    current_count = existing_total
    target_count = AUGMENT_TARGETS[class_label]
    needed = max(0, target_count - current_count)

    if needed == 0:
        if class_label == 1:
            print("\nBenign already at target, skipping")
            return 0
        print(
            f"\nClass {class_label} ({CLASS_NAMES[class_label]}) already has "
            f"≥{target_count} samples"
        )
        return 0

    print(f"\nAugmenting class {class_label} ({CLASS_NAMES[class_label]})")
    print(f"  Current: {current_count}, Target: {target_count}, Need: {needed}")

    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    augmented_count = 0
    aug_index = next_aug_index

    with tqdm(total=needed, desc=f"Class {class_label} augmentation", unit="sample") as pbar:
        while augmented_count < needed:
            original_path = random.choice(filepaths)
            volume = load_sample(original_path)

            if volume is None:
                continue

            aug_volume = apply_random_augmentations(volume)
            original_stem = original_path.stem
            aug_filename = f"aug_{original_stem}_{aug_index}.npy"
            aug_path = AUGMENTED_DIR / aug_filename

            try:
                np.save(aug_path, aug_volume)
                augmented_count += 1
                aug_index += 1
                pbar.update(1)

                if augmented_count % 10 == 0:
                    print(f"  Saved {augmented_count}/{needed} augmented samples")
            except Exception as exc:
                print(f"Error saving {aug_path}: {exc}")

    print(f"  ✓ Generated {augmented_count} augmented samples for class {class_label}")
    return augmented_count


def undersample_normal_class(normal_filepaths: list[Path]) -> list[Path]:
    """Randomly select UNDERSAMPLE_TARGET normal samples."""
    if len(normal_filepaths) <= UNDERSAMPLE_TARGET:
        print(
            f"\nNormal class has {len(normal_filepaths)} samples, "
            f"no undersampling needed (target: {UNDERSAMPLE_TARGET})"
        )
        return normal_filepaths

    print(f"\nUndersampling Normal class from {len(normal_filepaths)} to {UNDERSAMPLE_TARGET}")
    selected = random.sample(normal_filepaths, UNDERSAMPLE_TARGET)

    try:
        with open(NORMAL_SELECTED_FILE, "w") as f:
            for path in selected:
                f.write(str(path) + "\n")
        print(f"Saved selected normal samples to {NORMAL_SELECTED_FILE}")
    except Exception as exc:
        print(f"Error writing normal selection file: {exc}")

    return selected


def collect_all_filepaths(
    class_samples: dict[int, list[Path]],
    normal_selected: list[Path],
) -> tuple[list[dict], Counter]:
    """Collect all filepaths including originals and augmented."""
    all_records = []
    class_counts = Counter()

    print("\nCollecting all records (original + augmented)...")

    for label in sorted(CLASS_NAMES):
        if label == UNDERSAMPLE_CLASS:
            filepaths = normal_selected
        else:
            filepaths = class_samples.get(label, [])

        source_name = CLASS_NAMES[label].lower()

        for filepath in filepaths:
            all_records.append(
                {
                    "filepath": str(filepath),
                    "label": label,
                    "source": source_name,
                    "is_augmented": False,
                }
            )
            class_counts[label] += 1

    augmented_candidates: dict[int, list[Path]] = defaultdict(list)
    if AUGMENTED_DIR.exists():
        print("Adding augmented samples...")
        for aug_file in sorted(AUGMENTED_DIR.glob("aug_*.npy")):
            try:
                parts = aug_file.stem.split("_")
                if len(parts) < 2:
                    continue

                label = int(parts[-2])
                if label not in CLASS_NAMES:
                    continue

                augmented_candidates[label].append(aug_file)
            except (ValueError, IndexError):
                continue

    for label in sorted(augmented_candidates):
        files_for_label = augmented_candidates[label]

        if label in AUGMENT_TARGETS:
            needed = max(0, AUGMENT_TARGETS[label] - class_counts[label])
            selected_augmented = files_for_label[:needed]
        else:
            selected_augmented = files_for_label

        source_name = CLASS_NAMES[label].lower()
        for aug_file in selected_augmented:
            all_records.append(
                {
                    "filepath": str(aug_file),
                    "label": label,
                    "source": source_name,
                    "is_augmented": True,
                }
            )
            class_counts[label] += 1

    return all_records, class_counts


def regenerate_splits(all_records: list[dict]) -> None:
    """Regenerate train/val/test splits with manual index-based method."""
    print("\nRegenerating train/val/test splits...")

    df = pd.DataFrame(all_records)[["filepath", "label", "source", "is_augmented"]]
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(shuffled_df)
    train_end = int(0.70 * n)
    val_end = train_end + int(0.15 * n)

    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n  {split_name} split class distribution:")
        for label in sorted(CLASS_NAMES):
            count = len(split_df[split_df["label"] == label])
            aug_count = len(split_df[(split_df["label"] == label) & (split_df["is_augmented"])])
            if count > 0:
                print(f"    Class {label} ({CLASS_NAMES[label]}): {count} ({aug_count} augmented)")


def print_summary(
    before_counts: Counter,
    after_counts: Counter,
    aug_counts: dict[int, int],
) -> None:
    """Print final summary."""
    print("\n" + "=" * 80)
    print("AUGMENTATION SUMMARY")
    print("=" * 80)

    print("\nBefore augmentation:")
    for label in sorted(CLASS_NAMES):
        count = before_counts.get(label, 0)
        status = ""
        if label in AUGMENT_CLASSES:
            status = f" (target: {AUGMENT_TARGETS[label]})"
        elif label == UNDERSAMPLE_CLASS:
            status = f" (target: {UNDERSAMPLE_TARGET})"
        print(f"  Class {label} ({CLASS_NAMES[label]}): {count}{status}")

    print("\nAfter augmentation:")
    for label in sorted(CLASS_NAMES):
        count = after_counts.get(label, 0)
        target = AUGMENT_TARGETS.get(label, UNDERSAMPLE_TARGET if label == UNDERSAMPLE_CLASS else None)
        if target is not None:
            print(f"  Class {label} ({CLASS_NAMES[label]}): {count} (target: {target})")
        else:
            print(f"  Class {label} ({CLASS_NAMES[label]}): {count}")

    print(f"\nTotal final samples: {sum(after_counts.values())}")

    print("\nAugmented samples generated:")
    for label in sorted(AUGMENT_TARGETS):
        count = aug_counts.get(label, 0)
        if count > 0:
            print(f"  Class {label} ({CLASS_NAMES[label]}): {count} new samples")
        else:
            print(f"  Class {label} ({CLASS_NAMES[label]}): 0 new samples")

    print("\nOutput files:")
    print(f"  Augmented samples: {AUGMENTED_DIR}")
    print(f"  Train split: {SPLITS_DIR / 'train.csv'}")
    print(f"  Val split:   {SPLITS_DIR / 'val.csv'}")
    print(f"  Test split:  {SPLITS_DIR / 'test.csv'}")
    if NORMAL_SELECTED_FILE.exists():
        print(f"  Normal selection: {NORMAL_SELECTED_FILE}")


def augment_class(source_dir: Path, label: int, target_count: int, output_dir: Path) -> None:
    """Augment one class from one source directory until target_count is reached."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_files: list[Path] = []
    for npy_file in sorted(source_dir.rglob("*.npy")):
        parts = npy_file.stem.split("_")
        if not parts:
            continue

        file_label: int | None = None
        for idx in (-1, -2):
            try:
                file_label = int(parts[idx])
                break
            except (ValueError, IndexError):
                continue

        if file_label == label:
            matching_files.append(npy_file)

    real_count = len(matching_files)

    existing_aug_count = 0
    max_index = -1
    for aug_file in output_dir.glob("aug_*.npy"):
        parts = aug_file.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            aug_label = int(parts[-2])
            aug_index = int(parts[-1])
        except ValueError:
            continue

        if aug_label == label:
            existing_aug_count += 1
            max_index = max(max_index, aug_index)

    current_total = real_count + existing_aug_count
    print(f"Class {label}: real={real_count}, existing_augmented={existing_aug_count}, target={target_count}")

    if real_count == 0:
        print(f"No source files found for class {label} in {source_dir}")
        return

    if current_total >= target_count:
        print(f"Class {label} already at/above target ({current_total}/{target_count}), nothing to do.")
        return

    next_index = max_index + 1
    generated = 0
    needed = target_count - current_total

    print(f"Generating {needed} augmented samples for class {label}...")
    while current_total < target_count:
        original_path = random.choice(matching_files)
        try:
            volume = np.load(original_path).astype(np.float32)
        except Exception as exc:
            print(f"Failed to load {original_path}: {exc}")
            continue

        if volume.shape != (4, 128, 128, 128):
            print(f"Skipping {original_path}: expected shape (4, 128, 128, 128), got {volume.shape}")
            continue

        aug_volume = volume.copy()

        angle = random.uniform(-10.0, 10.0)
        for ch in range(4):
            aug_volume[ch] = ndimage.rotate(
                aug_volume[ch], angle, reshape=False, order=1, mode="nearest"
            )

        if random.random() < 0.5:
            flip_axis = random.choice([1, 2, 3])
            aug_volume = np.flip(aug_volume, axis=flip_axis).copy()

        scale = random.uniform(0.9, 1.1)
        aug_volume = aug_volume * scale

        noise = np.random.normal(0.0, 0.01, size=aug_volume.shape).astype(np.float32)
        aug_volume = np.clip(aug_volume + noise, 0, 1).astype(np.float32)

        out_name = f"aug_{original_path.stem}_{next_index}.npy"
        out_path = output_dir / out_name

        try:
            np.save(out_path, aug_volume)

            try:
                reloaded = np.load(out_path)
                valid_size = out_path.stat().st_size > 0
                valid_shape = reloaded.shape == (4, 128, 128, 128)
                valid_finite = np.isfinite(reloaded).all()
                channel_stds = [float(np.std(reloaded[ch])) for ch in range(4)]
                valid_std = all(std > 0.01 for std in channel_stds)
                channel_means = [float(np.mean(reloaded[ch])) for ch in range(4)]
                valid_mean = all(0.05 < mean < 0.95 for mean in channel_means)

                if not (valid_size and valid_shape and valid_finite and valid_std and valid_mean):
                    if out_path.exists():
                        os.remove(out_path)
                    print(f"Skipped corrupt augmented file: {out_path.name}")
                    continue
            except Exception:
                if out_path.exists():
                    os.remove(out_path)
                print(f"Skipped corrupt augmented file: {out_path.name}")
                continue

            next_index += 1
            generated += 1
            current_total += 1

            if generated % 10 == 0 or current_total == target_count:
                print(
                    f"Progress class {label}: generated={generated}, "
                    f"total={current_total}/{target_count}"
                )
        except Exception as exc:
            print(f"Failed to save {out_path}: {exc}")

    print(f"Done class {label}: generated {generated} files, total now {current_total}.")


def main() -> None:
    """Main augmentation pipeline."""
    print("=" * 80)
    print("DATA AUGMENTATION PIPELINE")
    print("=" * 80)

    set_seed()
    augment_class(
        source_dir=PROCESSED_DIR / "ms",
        label=4,
        target_count=200,
        output_dir=PROCESSED_DIR / "augmented",
    )
    augment_class(
        source_dir=PROCESSED_DIR / "lumiere",
        label=3,
        target_count=560,
        output_dir=PROCESSED_DIR / "augmented",
    )
    return

    set_seed()

    class_samples, before_counts = find_existing_samples()
    _, next_aug_indexes = summarize_existing_augmented_samples()

    existing_total_counts = Counter(before_counts)
    for label in AUGMENT_TARGETS:
        existing_total_counts[label] = count_total_existing_for_class(
            label,
            before_counts.get(label, 0),
        )

    aug_counts = {}
    for class_label in AUGMENT_TARGETS:
        filepaths = class_samples.get(class_label, [])
        if not filepaths:
            continue

        aug_count = augment_class_samples(
            class_label,
            filepaths,
            existing_total_counts.get(class_label, 0),
            next_aug_indexes.get(class_label, 0),
        )
        aug_counts[class_label] = aug_count

    normal_filepaths = class_samples.get(UNDERSAMPLE_CLASS, [])
    normal_selected = undersample_normal_class(normal_filepaths)

    # Ensure key classes are using corrected counts before split regeneration.
    class_samples[UNDERSAMPLE_CLASS] = normal_selected
    class_samples[0] = sorted(set(class_samples.get(0, [])), key=lambda p: str(p))

    print("\nCorrected counts before regenerating splits:")
    print(f"  Class 0 (Malignant): {existing_total_counts.get(0, 0)}")
    print(f"  Class 1 (Benign): {existing_total_counts.get(1, 0)}")
    print(f"  Class 2 (Normal selected): {len(class_samples.get(2, []))}")
    print(f"  Class 3 (Scar): {existing_total_counts.get(3, 0)}")
    print(f"  Class 4 (Inflammatory): {existing_total_counts.get(4, 0)}")

    all_records, after_counts = collect_all_filepaths(class_samples, normal_selected)

    regenerate_splits(all_records)

    print_summary(before_counts, after_counts, aug_counts)


if __name__ == "__main__":
    main()
