from pathlib import Path
import shutil


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] ensured: {path}")


def move_item(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[SKIP] source missing: {src}")
        return

    if dst.exists():
        print(f"[SKIP] destination exists: {dst}")
        return

    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))
    print(f"[MOVE] {src} -> {dst}")


def move_children(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"[SKIP] source directory missing: {src_dir}")
        return

    ensure_dir(dst_dir)
    for child in src_dir.iterdir():
        move_item(child, dst_dir / child.name)


def main() -> None:
    project_root = Path(__file__).resolve().parent

    # Target structure
    data_raw_brats_training = project_root / "data" / "raw" / "BraTS2020" / "training"
    data_raw_brats_validation = project_root / "data" / "raw" / "BraTS2020" / "validation"
    data_raw_remind_images = project_root / "data" / "raw" / "ReMIND" / "images"
    data_raw_remind_clinical = project_root / "data" / "raw" / "ReMIND" / "clinical_data.xlsx"
    data_raw_cai2r = project_root / "data" / "raw" / "CAI2R"
    data_processed_mri = project_root / "data" / "processed" / "MRI"
    data_processed_ct = project_root / "data" / "processed" / "CT"
    data_processed_pet = project_root / "data" / "processed" / "PET"
    data_splits = project_root / "data" / "splits"
    src_dir = project_root / "src"
    checkpoints_dir = project_root / "checkpoints"
    outputs_dir = project_root / "outputs"

    # Create all required directories
    required_dirs = [
        data_raw_brats_training,
        data_raw_brats_validation,
        data_raw_remind_images,
        data_raw_cai2r,
        data_processed_mri,
        data_processed_ct,
        data_processed_pet,
        data_splits,
        src_dir,
        checkpoints_dir,
        outputs_dir,
    ]
    for d in required_dirs:
        ensure_dir(d)

    # BraTS2020 moves: move contents of MICCAI folders
    brats_train_src = project_root / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    brats_val_src = project_root / "BraTS2020_ValidationData" / "MICCAI_BraTS2020_ValidationData"
    move_children(brats_train_src, data_raw_brats_training)
    move_children(brats_val_src, data_raw_brats_validation)

    # ReMIND images: move ReMIND-001 ... ReMIND-114 folders under remind/
    remind_src = project_root / "remind"
    if remind_src.exists() and remind_src.is_dir():
        for child in remind_src.iterdir():
            if child.is_dir() and child.name.startswith("ReMIND-"):
                move_item(child, data_raw_remind_images / child.name)
    else:
        print(f"[SKIP] source directory missing: {remind_src}")

    # ReMIND clinical Excel move and rename
    clinical_src = project_root / "Data" / "Raw" / "ReMIND-Dataset-Clinical-Data-September-2023.xlsx"
    move_item(clinical_src, data_raw_remind_clinical)

    # CAI2R MAT move
    cai2r_mat_src = project_root / "rawdata_mprage_fdg_2013.mat"
    move_item(cai2r_mat_src, data_raw_cai2r / "rawdata_mprage_fdg_2013.mat")

    print("Reorganization completed.")


if __name__ == "__main__":
    main()