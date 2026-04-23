"""Preprocess BraTS2020 and ReMIND into a unified .npy format.

Outputs:
- BraTS tumor samples: 4 x 128 x 128 x 128
- BraTS normal slices: 4 x 128 x 128
- ReMIND samples: 4 x 128 x 128 x 128

The script processes one patient at a time, saves arrays to disk, and builds
stratified train/val/test CSV splits from the saved files.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_MODE = False
TARGET_SHAPE = (128, 128, 128)
OUTPUT_DIR = PROJECT_ROOT / "Data" / "processed" / "MRI"
BASE_DIR = PROJECT_ROOT
PROCESSED_DIR = OUTPUT_DIR
SPLITS_DIR = PROJECT_ROOT / "Data" / "splits"
BRATS_TRAIN = PROJECT_ROOT / "Data" / "Raw" / "BraTS2020" / "training"
BRATS_MAPPING = PROJECT_ROOT / "Data" / "Raw" / "BraTS2020" / "training" / "name_mapping.csv"
REMIND_IMAGES = PROJECT_ROOT / "Data" / "Raw" / "ReMIND" / "images"
REMIND_CLINICAL = PROJECT_ROOT / "Data" / "Raw" / "ReMIND" / "clinical_data.xlsx"
IXI_T1_DIR = BASE_DIR / "Data" / "Raw" / "IXI" / "T1"
IXI_T2_DIR = BASE_DIR / "Data" / "Raw" / "IXI" / "T2"
IXI_OUT_DIR = PROCESSED_DIR / "ixi"
IXI_TARGET = 361
IXI_SEED = 42
IXI_LABEL = 2

CLASS_NAMES = {
    0: "Malignant",
    1: "Benign",
    2: "Normal",
    3: "Scar",
    4: "Inflammatory",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def parse_grade(value: object) -> str:
    return normalize_text(value).lower()


def format_remind_case_id(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if text.lower().startswith("remind-"):
        return text
    if text.isdigit():
        return f"ReMIND-{int(text):03d}"
    try:
        numeric_value = int(float(text))
        return f"ReMIND-{numeric_value:03d}"
    except ValueError:
        return text


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int], order: int) -> np.ndarray:
    zoom_factors = [target / current for target, current in zip(target_shape, volume.shape)]
    resized = zoom(volume, zoom_factors, order=order)
    resized = np.asarray(resized)

    if resized.shape != target_shape:
        corrected = np.zeros(target_shape, dtype=resized.dtype)
        slices = tuple(slice(0, min(dim, target)) for dim, target in zip(resized.shape, target_shape))
        corrected[slices] = resized[slices]
        resized = corrected

    return resized


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    lower = np.percentile(volume, 1)
    upper = np.percentile(volume, 99)
    clipped = np.clip(volume, lower, upper)
    min_value = clipped.min()
    max_value = clipped.max()
    if math.isclose(float(max_value), float(min_value)):
        return np.zeros_like(clipped, dtype=np.float32)
    normalized = (clipped - min_value) / (max_value - min_value)
    return normalized.astype(np.float32)


def preprocess_modality(volume: np.ndarray) -> np.ndarray:
    resized = resize_volume(volume, TARGET_SHAPE, order=1)
    return normalize_volume(resized)


def preprocess_segmentation(volume: np.ndarray) -> np.ndarray:
    return resize_volume(volume, TARGET_SHAPE, order=0)


def save_npy(array: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    np.save(output_path, array.astype(np.float32))


def label_from_output_filename(path: Path) -> Optional[int]:
    stem = path.stem
    parts = stem.split("_")
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def build_brats_mapping(mapping_csv: Path) -> Dict[str, int]:
    mapping_df = pd.read_csv(mapping_csv)
    required_columns = {"Grade", "BraTS_2020_subject_ID"}
    missing = required_columns - set(mapping_df.columns)
    if missing:
        raise ValueError(f"BraTS mapping is missing columns: {sorted(missing)}")

    patient_to_label: Dict[str, int] = {}
    for _, row in mapping_df.iterrows():
        grade = normalize_text(row["Grade"]).upper()
        patient_id = normalize_text(row["BraTS_2020_subject_ID"])
        if not patient_id:
            continue
        if grade == "HGG":
            patient_to_label[patient_id] = 0
        elif grade == "LGG":
            patient_to_label[patient_id] = 1
    return patient_to_label


def get_nifti_data(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj)


def list_brats_patient_dirs(training_dir: Path) -> List[Path]:
    return sorted([path for path in training_dir.iterdir() if path.is_dir() and path.name.startswith("BraTS20_Training_")])


def print_braTS_path_preview(training_dir: Path, patient_dirs: List[Path]) -> None:
    print(f"Resolved BraTS training path: {training_dir.resolve()}")
    print("First 5 BraTS subfolders:")
    if not patient_dirs:
        print("  (none found)")
        return
    for patient_dir in patient_dirs[:5]:
        print(f"  - {patient_dir.name}")


def extract_brats_normal_slices(volume_4d: np.ndarray, seg_volume: np.ndarray, max_slices: int = 10) -> List[Tuple[int, np.ndarray]]:
    eligible_indices = np.where(np.all(seg_volume == 0, axis=(0, 1)))[0].tolist()
    if not eligible_indices:
        return []

    if len(eligible_indices) <= max_slices:
        selected_indices = eligible_indices
    else:
        positions = np.linspace(0, len(eligible_indices) - 1, max_slices)
        selected_indices = [eligible_indices[int(round(pos))] for pos in positions]

    normal_slices: List[Tuple[int, np.ndarray]] = []
    seen = set()
    for idx in selected_indices:
        if idx in seen:
            continue
        seen.add(idx)
        slice_array = volume_4d[:, :, :, idx]
        normal_slices.append((idx, slice_array))
    return normal_slices


def process_brats() -> Tuple[List[dict], Counter, List[str], dict]:
    print("\n" + "=" * 80)
    print("PART 1 - BraTS2020 preprocessing")
    print("=" * 80)

    output_dir = OUTPUT_DIR / "brats"
    ensure_dir(output_dir)

    patient_to_label = build_brats_mapping(BRATS_MAPPING)
    if not BRATS_TRAIN.exists():
        raise FileNotFoundError(f"BraTS training directory not found: {BRATS_TRAIN}")
    patient_dirs = list_brats_patient_dirs(BRATS_TRAIN)
    print_braTS_path_preview(BRATS_TRAIN, patient_dirs)

    records: List[dict] = []
    class_counts: Counter = Counter()
    skipped: List[str] = []
    stats = {"attempted": 0, "succeeded": 0, "failed": 0}

    print(f"BraTS patients found: {len(patient_dirs)}")
    for index, patient_dir in enumerate(tqdm(patient_dirs, desc="BraTS patients", unit="patient"), start=1):
        if DEBUG_MODE and index > 5:
            print("DEBUG_MODE is enabled: stopping BraTS after first 5 patients.")
            break

        patient_id = patient_dir.name
        try:
            if DEBUG_MODE:
                print(f"\n[DEBUG][BraTS] Processing patient {index}: {patient_id}")
                print(f"[DEBUG][BraTS] Patient directory: {patient_dir}")

            stats["attempted"] += 1

            mapped_label = patient_to_label.get(patient_id)
            if mapped_label is None:
                skipped.append(f"BraTS {patient_id}: no label in name_mapping.csv")
                stats["failed"] += 1
                continue

            tumor_path = output_dir / f"brats_{patient_id}_{mapped_label}.npy"
            if tumor_path.exists():
                tqdm.write(f"BraTS {patient_id}: Skipped (already exists)")
                existing_label = label_from_output_filename(tumor_path)
                if existing_label is not None:
                    records.append({"filepath": str(tumor_path), "label": existing_label, "source": "brats"})
                    class_counts[existing_label] += 1

                    normal_pattern = f"brats_{patient_id}_normal_*.npy"
                    for normal_path in sorted(output_dir.glob(normal_pattern)):
                        records.append({"filepath": str(normal_path), "label": 2, "source": "brats"})
                        class_counts[2] += 1

                stats["succeeded"] += 1
                continue

            t1_path = patient_dir / f"{patient_id}_t1.nii"
            t1ce_path = patient_dir / f"{patient_id}_t1ce.nii"
            t2_path = patient_dir / f"{patient_id}_t2.nii"
            flair_path = patient_dir / f"{patient_id}_flair.nii"
            seg_path = patient_dir / f"{patient_id}_seg.nii"

            required_files = [t1_path, t1ce_path, t2_path, flair_path, seg_path]
            missing_files = [str(path.name) for path in required_files if not path.exists()]
            if missing_files:
                skipped.append(f"BraTS {patient_id}: missing files {missing_files}")
                stats["failed"] += 1
                continue

            t1 = get_nifti_data(t1_path)
            t1ce = get_nifti_data(t1ce_path)
            t2 = get_nifti_data(t2_path)
            flair = get_nifti_data(flair_path)
            seg = get_nifti_data(seg_path)

            if not np.any(seg > 0):
                skipped.append(f"BraTS {patient_id}: segmentation mask has no non-zero labels")
                stats["failed"] += 1
                continue

            modalities = [t1, t1ce, t2, flair]
            processed_modalities = [preprocess_modality(modality) for modality in modalities]
            processed_seg = preprocess_segmentation(seg)
            stacked = np.stack(processed_modalities, axis=0).astype(np.float32)

            if DEBUG_MODE:
                print(f"[DEBUG][BraTS] T1 shape: {t1.shape}, T1CE shape: {t1ce.shape}, T2 shape: {t2.shape}, FLAIR shape: {flair.shape}")
                print(f"[DEBUG][BraTS] Segmentation unique labels: {sorted(np.unique(seg).astype(int).tolist())}")
                print(f"[DEBUG][BraTS] Final tumor sample shape: {stacked.shape}")

            label = mapped_label
            save_npy(stacked, tumor_path)
            records.append({"filepath": str(tumor_path), "label": label, "source": "brats"})
            class_counts[label] += 1
            stats["succeeded"] += 1

            if DEBUG_MODE:
                print(f"[DEBUG][BraTS] Saved tumor sample: {tumor_path}")

            normal_slices = extract_brats_normal_slices(stacked, processed_seg, max_slices=10)
            for slice_idx, slice_array in normal_slices:
                normal_path = output_dir / f"brats_{patient_id}_normal_{slice_idx}.npy"
                save_npy(slice_array, normal_path)
                records.append({"filepath": str(normal_path), "label": 2, "source": "brats"})
                class_counts[2] += 1

                if DEBUG_MODE:
                    print(f"[DEBUG][BraTS] Saved normal slice {slice_idx}: {normal_path}")

            if index % 10 == 0:
                tqdm.write(f"BraTS progress: processed {index}/{len(patient_dirs)} patients")
        except Exception as exc:
            skipped.append(f"BraTS {patient_id}: {type(exc).__name__} - {exc}")
            stats["failed"] += 1

    return records, class_counts, skipped, stats


def map_remind_label(histopathology: object, grade_value: object) -> Optional[int]:
    hist = normalize_text(histopathology)
    hist_lower = hist.lower()
    grade = parse_grade(grade_value)

    # Histopathology drives mapping. WHO grade is used only for astrocytoma tie-breaking.
    if not hist_lower:
        return None

    if "effects of treatment" in hist_lower or "necrosis" in hist_lower:
        return 3
    if hist_lower == "reactive brain and chronic inflammation":
        return 3

    if (
        hist_lower == "non-tumor destructive chronic inflammatory lesion with abnormal vasculature"
        or "inflammatory" in hist_lower
        or "inflammation" in hist_lower
        or "demyelinating lesions" in hist_lower
    ):
        return 4

    if hist_lower == "non-tumor epileptogenic brain parenchyma and gray matter":
        return 2

    if "metastatic" in hist_lower or "metastastic" in hist_lower:
        return 0
    if hist_lower == "glioblastoma":
        return 0
    if hist_lower == "primary diffuse large b-cell lymphoma":
        return 0
    if hist_lower == "atypical meningioma":
        return 0
    if hist_lower == "glioma with ependymal features":
        return 0

    if hist_lower == "oligodendroglioma":
        return 1
    if hist_lower == "low grade glioma":
        return 1
    if hist_lower == "dysembryoplastic neuroepithelial tumor":
        return 1
    if hist_lower == "papillary glioneuronal tumor":
        return 1
    if hist_lower == "glioneuronal tumor":
        return 1
    if hist_lower == "glial fibroma":
        return 1
    if "hypercellular brain tissue with scattered atypical cells" in hist_lower:
        return 1

    if hist_lower == "astrocytoma":
        if grade in {"3", "4"}:
            return 0
        if grade in {"1", "2"}:
            return 1
        # If grade is missing or not assigned, keep a deterministic default.
        return 1

    return None


def get_primary_series(dcm_files: Sequence[Path]) -> List[Path]:
    grouped: Dict[Path, List[Path]] = defaultdict(list)
    for file_path in dcm_files:
        grouped[file_path.parent].append(file_path)

    best_series = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), str(item[0]))
    )[0][1]
    return sorted(best_series, key=lambda path: path.name)


def process_remind() -> Tuple[List[dict], Counter, List[str], dict]:
    print("\n" + "=" * 80)
    print("PART 2 - ReMIND preprocessing")
    print("=" * 80)

    output_dir = OUTPUT_DIR / "remind"
    ensure_dir(output_dir)

    clinical_df = pd.read_excel(REMIND_CLINICAL)
    required_columns = {"Case Number", "Histopathology", "WHO Grade"}
    missing = required_columns - set(clinical_df.columns)
    if missing:
        raise ValueError(f"ReMIND clinical file is missing columns: {sorted(missing)}")

    label_map: Dict[str, int] = {}
    clinical_lookup: Dict[str, dict] = {}
    for _, row in clinical_df.iterrows():
        case_id = format_remind_case_id(row["Case Number"])
        label = map_remind_label(row["Histopathology"], row["WHO Grade"])
        clinical_lookup[case_id] = {
            "Histopathology": normalize_text(row["Histopathology"]),
            "WHO Grade": normalize_text(row["WHO Grade"]),
        }
        if label is None:
            continue
        label_map[case_id] = label

    patient_dirs = sorted([path for path in REMIND_IMAGES.iterdir() if path.is_dir()])
    records: List[dict] = []
    class_counts: Counter = Counter()
    skipped: List[str] = []
    debug_mapping_rows: List[dict] = []
    stats = {"attempted": 0, "succeeded": 0, "failed": 0}

    print(f"ReMIND patients found: {len(patient_dirs)}")
    for index, patient_dir in enumerate(tqdm(patient_dirs, desc="ReMIND patients", unit="patient"), start=1):
        if DEBUG_MODE and index > 5:
            print("DEBUG_MODE is enabled: stopping ReMIND after first 5 patients.")
            break

        case_id = patient_dir.name
        try:
            if DEBUG_MODE:
                print(f"\n[DEBUG][ReMIND] Processing patient {index}: {case_id}")

            stats["attempted"] += 1

            clinical_info = clinical_lookup.get(case_id, {})
            hist_value = clinical_info.get("Histopathology", "")
            grade_value = clinical_info.get("WHO Grade", "")
            patient_label = label_map.get(case_id)
            if patient_label is None:
                reason = f"unrecognized histopathology (Histopathology={hist_value!r}, WHO Grade={grade_value!r})"
                skipped.append(f"ReMIND {case_id}: {reason}")
                stats["failed"] += 1
                if DEBUG_MODE:
                    debug_mapping_rows.append({
                        "Case ID": case_id,
                        "Histopathology": hist_value,
                        "WHO Grade": grade_value,
                        "Mapped class label": "SKIPPED",
                        "Reason": reason,
                    })
                continue

            output_path = output_dir / f"remind_{case_id}_{patient_label}.npy"
            if output_path.exists():
                tqdm.write(f"ReMIND {case_id}: Skipped (already exists)")
                existing_label = label_from_output_filename(output_path)
                if existing_label is not None:
                    records.append({"filepath": str(output_path), "label": existing_label, "source": "remind"})
                    class_counts[existing_label] += 1
                stats["succeeded"] += 1
                continue

            if DEBUG_MODE:
                debug_mapping_rows.append({
                    "Case ID": case_id,
                    "Histopathology": hist_value,
                    "WHO Grade": grade_value,
                    "Mapped class label": patient_label,
                    "Reason": "mapped successfully",
                })

            dcm_files = sorted(patient_dir.rglob("*.dcm"), key=lambda path: path.name)
            if not dcm_files:
                skipped.append(f"ReMIND {case_id}: no DICOM files found")
                stats["failed"] += 1
                continue

            primary_series_files = get_primary_series(dcm_files)
            slices = []
            for dcm_path in primary_series_files:
                ds = pydicom.dcmread(str(dcm_path))
                slices.append(np.asarray(ds.pixel_array))

            if DEBUG_MODE:
                print(f"[DEBUG][ReMIND] DICOM files in primary series: {len(primary_series_files)}")
                print(f"[DEBUG][ReMIND] First DICOM: {primary_series_files[0].name}")

            if not slices:
                skipped.append(f"ReMIND {case_id}: primary series could not be loaded")
                stats["failed"] += 1
                continue

            volume = np.stack(slices, axis=-1)
            if volume.ndim != 3:
                skipped.append(f"ReMIND {case_id}: unexpected DICOM volume shape {volume.shape}")
                stats["failed"] += 1
                continue

            resized = resize_volume(volume, TARGET_SHAPE, order=1)
            normalized = normalize_volume(resized)
            stacked = np.stack([normalized] * 4, axis=0).astype(np.float32)

            if DEBUG_MODE:
                print(f"[DEBUG][ReMIND] Raw volume shape: {volume.shape}")
                print(f"[DEBUG][ReMIND] Resized volume shape: {resized.shape}")
                print(f"[DEBUG][ReMIND] Final stacked sample shape: {stacked.shape}")
                print(f"[DEBUG][ReMIND] Assigned label: {patient_label}")

            save_npy(stacked, output_path)
            records.append({"filepath": str(output_path), "label": patient_label, "source": "remind"})
            class_counts[patient_label] += 1
            stats["succeeded"] += 1

            if DEBUG_MODE:
                print(f"[DEBUG][ReMIND] Saved sample: {output_path}")

            if index % 10 == 0:
                tqdm.write(f"ReMIND progress: processed {index}/{len(patient_dirs)} patients")
        except Exception as exc:
            skipped.append(f"ReMIND {case_id}: {type(exc).__name__} - {exc}")
            stats["failed"] += 1

    if DEBUG_MODE and not records:
        print("\n[DEBUG][ReMIND] Clinical mapping preview for first processed patients:")
        if debug_mapping_rows:
            debug_df = pd.DataFrame(debug_mapping_rows[:5])
            print(debug_df.to_string(index=False))
        else:
            print("  No clinical mapping rows were captured.")

    return records, class_counts, skipped, stats


def process_ixi() -> Tuple[List[dict], Counter, List[str], dict]:
    print("\n" + "=" * 80)
    print("PART 3 - IXI normal preprocessing")
    print("=" * 80)

    if not IXI_T1_DIR.exists():
        raise FileNotFoundError(f"IXI T1 directory not found: {IXI_T1_DIR}")
    if not IXI_T2_DIR.exists():
        raise FileNotFoundError(f"IXI T2 directory not found: {IXI_T2_DIR}")

    ensure_dir(IXI_OUT_DIR)

    t1_files = sorted(IXI_T1_DIR.glob("IXI*-T1.nii.gz"))
    t2_lookup: Dict[str, Path] = {}
    for t2_path in sorted(IXI_T2_DIR.glob("IXI*-T2.nii.gz")):
        t2_name = t2_path.name
        if not t2_name.endswith("-T2.nii.gz"):
            continue
        t2_base = t2_name[: -len("-T2.nii.gz")]
        t2_subject_id = t2_base.split("-")[0]
        t2_lookup.setdefault(t2_subject_id, t2_path)

    valid_pairs: List[Tuple[Path, Path, str]] = []
    for t1_path in t1_files:
        t1_name = t1_path.name
        if not t1_name.endswith("-T1.nii.gz"):
            continue
        t1_base = t1_name[: -len("-T1.nii.gz")]
        subject_id = t1_base.split("-")[0]
        matching_t2 = t2_lookup.get(subject_id)
        if matching_t2 is not None:
            valid_pairs.append((t1_path, matching_t2, subject_id))

    valid_pairs = sorted(valid_pairs, key=lambda item: (item[2], item[0].name, item[1].name))
    sample_count = min(IXI_TARGET, len(valid_pairs))
    selected_pairs = random.Random(IXI_SEED).sample(valid_pairs, sample_count) if sample_count > 0 else []

    records: List[dict] = []
    skipped: List[str] = []
    succeeded = 0
    failed = 0
    attempted = len(selected_pairs)

    for t1_path, t2_path, subject_id in tqdm(selected_pairs, desc="IXI Normal", unit="case"):
        try:
            t1 = nib.load(str(t1_path)).get_fdata().astype(np.float32)
            t2 = nib.load(str(t2_path)).get_fdata().astype(np.float32)

            t1 = resize_volume(t1, TARGET_SHAPE, order=1)
            t2 = resize_volume(t2, TARGET_SHAPE, order=1)

            t1 = normalize_volume(t1)
            t2 = normalize_volume(t2)

            stacked = np.stack([t1, t1, t2, t2], axis=0).astype(np.float32)
            out_path = IXI_OUT_DIR / f"ixi_{subject_id}_{IXI_LABEL}.npy"
            np.save(out_path, stacked)

            records.append({
                "filepath": str(out_path),
                "label": IXI_LABEL,
                "source": "ixi",
                "is_augmented": False,
            })
            succeeded += 1
        except Exception as exc:
            skipped.append(f"IXI {subject_id}: {type(exc).__name__} - {exc}")
            failed += 1

    return records, Counter({IXI_LABEL: succeeded}), skipped, {"attempted": attempted, "succeeded": succeeded, "failed": failed}


def process_lumiere() -> Tuple[List[dict], Counter, List[str], dict]:
    print("\n" + "=" * 80)
    print("PART 4 - Lumiere scar preprocessing")
    print("=" * 80)

    imaging_root = PROJECT_ROOT / "Scar dataset" / "Imaging"
    output_dir = OUTPUT_DIR / "lumiere"

    if not imaging_root.exists():
        raise FileNotFoundError(f"Lumiere imaging directory not found: {imaging_root}")

    ensure_dir(output_dir)

    patient_dirs = sorted([path for path in imaging_root.iterdir() if path.is_dir()])
    records: List[dict] = []
    skipped: List[str] = []
    succeeded = 0
    failed = 0
    attempted = 0

    for patient_index, patient_dir in enumerate(patient_dirs, start=1):
        print(f"Lumiere patient {patient_index}/{len(patient_dirs)}: {patient_dir.name}")

        week_dirs = sorted([path for path in patient_dir.iterdir() if path.is_dir()])
        for week_dir in week_dirs:
            attempted += 1
            t1_path = week_dir / "T1.nii.gz"
            t1ce_path = week_dir / "CT1.nii.gz"
            t2_path = week_dir / "T2.nii.gz"
            flair_path = week_dir / "FLAIR.nii.gz"

            required_files = [t1_path, t1ce_path, t2_path, flair_path]
            if any(not path.exists() for path in required_files):
                missing = [path.name for path in required_files if not path.exists()]
                skipped.append(f"Lumiere {patient_dir.name}/{week_dir.name}: missing files {missing}")
                failed += 1
                continue

            try:
                t1 = get_nifti_data(t1_path)
                t1ce = get_nifti_data(t1ce_path)
                t2 = get_nifti_data(t2_path)
                flair = get_nifti_data(flair_path)

                t1 = normalize_volume(resize_volume(t1, TARGET_SHAPE, order=1))
                t1ce = normalize_volume(resize_volume(t1ce, TARGET_SHAPE, order=1))
                t2 = normalize_volume(resize_volume(t2, TARGET_SHAPE, order=1))
                flair = normalize_volume(resize_volume(flair, TARGET_SHAPE, order=1))

                stacked = np.stack([t1, t1ce, t2, flair], axis=0).astype(np.float32)
                out_path = output_dir / f"lumiere_{patient_dir.name}_{week_dir.name}_3.npy"
                save_npy(stacked, out_path)

                records.append({
                    "filepath": str(out_path),
                    "label": 3,
                    "source": "lumiere",
                    "is_augmented": False,
                })
                succeeded += 1
            except Exception as exc:
                skipped.append(f"Lumiere {patient_dir.name}/{week_dir.name}: {type(exc).__name__} - {exc}")
                failed += 1

    return records, Counter({3: succeeded}), skipped, {"attempted": attempted, "succeeded": succeeded, "failed": failed}


def process_ms() -> Tuple[List[dict], Counter, List[str], dict]:
    print("\n" + "=" * 80)
    print("PART 5 - MS inflammatory preprocessing")
    print("=" * 80)

    ms_root = PROJECT_ROOT / "Inflammatory dataset"
    output_dir = OUTPUT_DIR / "ms"

    if not ms_root.exists():
        raise FileNotFoundError(f"Inflammatory dataset directory not found: {ms_root}")

    ensure_dir(output_dir)

    patient_dirs = sorted([path for path in ms_root.iterdir() if path.is_dir() and path.name.startswith("Patient-")])
    records: List[dict] = []
    skipped: List[str] = []
    succeeded = 0
    failed = 0
    attempted = len(patient_dirs)

    for patient_index, patient_dir in enumerate(patient_dirs, start=1):
        print(f"MS patient {patient_index}/{len(patient_dirs)}: {patient_dir.name}")

        patient_suffix = patient_dir.name.split("-", 1)[1] if "-" in patient_dir.name else patient_dir.name
        t1_path = patient_dir / f"{patient_suffix}-T1.nii"
        t2_path = patient_dir / f"{patient_suffix}-T2.nii"
        flair_path = patient_dir / f"{patient_suffix}-Flair.nii"

        required_files = [t1_path, t2_path, flair_path]
        if any(not path.exists() for path in required_files):
            missing = [path.name for path in required_files if not path.exists()]
            skipped.append(f"MS {patient_dir.name}: missing files {missing}")
            failed += 1
            continue

        try:
            t1 = get_nifti_data(t1_path)
            t2 = get_nifti_data(t2_path)
            flair = get_nifti_data(flair_path)

            t1 = normalize_volume(resize_volume(t1, TARGET_SHAPE, order=1))
            t2 = normalize_volume(resize_volume(t2, TARGET_SHAPE, order=1))
            flair = normalize_volume(resize_volume(flair, TARGET_SHAPE, order=1))

            stacked = np.stack([t1, t1, t2, flair], axis=0).astype(np.float32)
            out_path = output_dir / f"ms_{patient_suffix}_4.npy"
            save_npy(stacked, out_path)

            records.append({
                "filepath": str(out_path),
                "label": 4,
                "source": "ms",
                "is_augmented": False,
            })
            succeeded += 1
        except Exception as exc:
            skipped.append(f"MS {patient_dir.name}: {type(exc).__name__} - {exc}")
            failed += 1

    return records, Counter({4: succeeded}), skipped, {"attempted": attempted, "succeeded": succeeded, "failed": failed}


def stratified_split(records: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not records:
        raise ValueError("No records were produced, so splits cannot be created.")

    df = pd.DataFrame(records)[["filepath", "label", "source"]]
    labels = df["label"]
    label_counts = labels.value_counts()
    insufficient_classes = [int(label) for label, count in label_counts.items() if int(count) < 2]
    # Two-stage stratification (70/15/15 via 70/30 then 50/50) needs enough samples
    # for each class to appear at least twice in the temporary 30% split.
    low_for_two_stage = [int(label) for label, count in label_counts.items() if int(count) < 7]
    underrepresented_classes = [int(label) for label, count in label_counts.items() if int(count) < 10]

    if underrepresented_classes:
        warning = ", ".join(
            f"{label} ({CLASS_NAMES.get(label, 'Unknown')}): {int(label_counts.get(label, 0))}"
            for label in sorted(underrepresented_classes)
        )
        print(f"\nData warning: underrepresented classes (<10 samples): {warning}")

    use_non_stratified = bool(insufficient_classes or low_for_two_stage)

    if insufficient_classes:
        class_summary = ", ".join(
            f"{label} ({CLASS_NAMES.get(label, 'Unknown')}): {int(label_counts.get(label, 0))}"
            for label in sorted(insufficient_classes)
        )
        print(
            "\nWarning: using manual index split because the following classes have fewer than 2 samples: "
            f"{class_summary}"
        )
    elif low_for_two_stage:
        class_summary = ", ".join(
            f"{label} ({CLASS_NAMES.get(label, 'Unknown')}): {int(label_counts.get(label, 0))}"
            for label in sorted(low_for_two_stage)
        )
        print(
            "\nWarning: using manual index split because two-stage stratified 70/15/15 is not feasible for low-count classes: "
            f"{class_summary}"
        )

    if use_non_stratified:
        shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(shuffled_df)
        train_end = int(0.70 * n)
        val_end = train_end + int(0.15 * n)

        train_df = shuffled_df.iloc[:train_end]
        val_df = shuffled_df.iloc[train_end:val_end]
        test_df = shuffled_df.iloc[val_end:]
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=42,
            shuffle=True,
            stratify=labels,
        )

        temp_labels = temp_df["label"]
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=42,
            shuffle=True,
            stratify=temp_labels,
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def print_split_distribution(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name} split: {len(df)} samples")
    counts = df["label"].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  Class {label} ({CLASS_NAMES.get(int(label), 'Unknown')}): {count}")


def save_split_csvs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    ensure_dir(SPLITS_DIR)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)


def print_summary(all_records: List[dict], class_counts: Counter, skipped: List[str], train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, ixi_stats: dict) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    print("\nTotal samples per class:")
    for label in sorted(CLASS_NAMES):
        print(f"  {label} ({CLASS_NAMES[label]}): {class_counts.get(label, 0)}")

    print("\nTotal samples per source:")
    source_counts = Counter(record["source"] for record in all_records)
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    print("\nTrain/Val/Test counts:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    print("\nAttempted / succeeded / failed by source:")
    print(f"  IXI: attempted={ixi_stats['attempted']}, succeeded={ixi_stats['succeeded']}, failed={ixi_stats['failed']}")

    print_split_distribution("Train", train_df)
    print_split_distribution("Validation", val_df)
    print_split_distribution("Test", test_df)

    print("\nSkipped patients:")
    if not skipped:
        print("  None")
    else:
        reason_counts = Counter()
        for item in skipped:
            reason = item.split(": ", 1)[1] if ": " in item else item
            reason_counts[reason] += 1
        for reason, count in reason_counts.most_common():
            print(f"  {count} - {reason}")


def print_graceful_split_block(all_records: List[dict], class_counts: Counter, skipped: List[str], brats_stats: dict, remind_stats: dict, ixi_stats: dict) -> None:
    print("\nNo valid stratified split could be created, so the script is exiting gracefully.")
    print(f"BraTS attempted: {brats_stats['attempted']}")
    print(f"ReMIND attempted: {remind_stats['attempted']}")
    print("\nAttempted / succeeded / failed by source:")
    print(f"  BraTS: attempted={brats_stats['attempted']}, succeeded={brats_stats['succeeded']}, failed={brats_stats['failed']}")
    print(f"  ReMIND: attempted={remind_stats['attempted']}, succeeded={remind_stats['succeeded']}, failed={remind_stats['failed']}")
    print(f"  IXI: attempted={ixi_stats['attempted']}, succeeded={ixi_stats['succeeded']}, failed={ixi_stats['failed']}")

    print("\nTotal samples per class:")
    for label in sorted(CLASS_NAMES):
        print(f"  {label} ({CLASS_NAMES[label]}): {class_counts.get(label, 0)}")

    print("\nTotal samples per source:")
    source_counts = Counter(record["source"] for record in all_records)
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    print("\nSkipped patients:")
    if not skipped:
        print("  None")
    else:
        for item in skipped:
            print(f"  - {item}")

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ixi-only", action="store_true", help="Run only IXI preprocessing")
    parser.add_argument("--brats-only", action="store_true", help="Run only BraTS preprocessing")
    parser.add_argument("--remind-only", action="store_true", help="Run only ReMIND preprocessing")
    parser.add_argument("--lumiere-only", action="store_true", help="Run only Lumiere scar preprocessing")
    parser.add_argument("--ms-only", action="store_true", help="Run only MS inflammatory preprocessing")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)
    ensure_dir(SPLITS_DIR)

    all_records: List[dict] = []
    class_counts: Counter = Counter()
    skipped: List[str] = []

    only_flags = [args.ixi_only, args.brats_only, args.remind_only, args.lumiere_only, args.ms_only]
    run_all = not any(only_flags)

    brats_stats = {"attempted": 0, "succeeded": 0, "failed": 0}
    remind_stats = {"attempted": 0, "succeeded": 0, "failed": 0}
    ixi_stats = {"attempted": 0, "succeeded": 0, "failed": 0}

    if args.brats_only or run_all:
        try:
            brats_records, brats_counts, brats_skipped, brats_stats = process_brats()
            all_records.extend(brats_records)
            class_counts.update(brats_counts)
            skipped.extend(brats_skipped)
        except Exception as exc:
            print(f"BraTS preprocessing failed: {type(exc).__name__} - {exc}")

    if args.remind_only or run_all:
        try:
            remind_records, remind_counts, remind_skipped, remind_stats = process_remind()
            all_records.extend(remind_records)
            class_counts.update(remind_counts)
            skipped.extend(remind_skipped)
        except Exception as exc:
            print(f"ReMIND preprocessing failed: {type(exc).__name__} - {exc}")

    if args.ixi_only or run_all:
        try:
            ixi_records, ixi_counts, ixi_skipped, ixi_stats = process_ixi()
            all_records.extend(ixi_records)
            class_counts.update(ixi_counts)
            skipped.extend(ixi_skipped)
        except Exception as exc:
            print(f"IXI preprocessing failed: {type(exc).__name__} - {exc}")

    if args.lumiere_only or run_all:
        try:
            lumiere_records, lumiere_counts, lumiere_skipped, _ = process_lumiere()
            all_records.extend(lumiere_records)
            class_counts.update(lumiere_counts)
            skipped.extend(lumiere_skipped)
        except Exception as exc:
            print(f"Lumiere preprocessing failed: {type(exc).__name__} - {exc}")

    if args.ms_only or run_all:
        try:
            ms_records, ms_counts, ms_skipped, _ = process_ms()
            all_records.extend(ms_records)
            class_counts.update(ms_counts)
            skipped.extend(ms_skipped)
        except Exception as exc:
            print(f"MS preprocessing failed: {type(exc).__name__} - {exc}")

    if not all_records:
        print_graceful_split_block(all_records, class_counts, skipped, brats_stats, remind_stats, ixi_stats)
        return

    try:
        train_df, val_df, test_df = stratified_split(all_records)
    except ValueError as exc:
        print(f"\nStratified split could not be created: {exc}")
        print_graceful_split_block(all_records, class_counts, skipped, brats_stats, remind_stats, ixi_stats)
        return

    save_split_csvs(train_df, val_df, test_df)
    print_summary(all_records, class_counts, skipped, train_df, val_df, test_df, ixi_stats)


if __name__ == "__main__":
    main()