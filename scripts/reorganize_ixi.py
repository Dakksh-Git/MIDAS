"""Reorganize IXI files into the project raw-data structure.

Moves:
- IXI*-T1.nii.gz -> Data/Raw/IXI/T1/
- IXI*-T2.nii.gz -> Data/Raw/IXI/T2/
- IXI.xls or IXI.xlsx -> Data/Raw/IXI/IXI.xls (or IXI.xlsx if .xls unavailable)
"""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
IXI_ROOT = PROJECT_ROOT / "Data" / "Raw" / "IXI"
T1_DEST = IXI_ROOT / "T1"
T2_DEST = IXI_ROOT / "T2"

SKIP_DIR_NAMES = {".venv"}
SKIP_DIR_PARTS = {"Data", "processed"}
PROGRESS_INTERVAL = 50


def is_excluded(path: Path) -> bool:
    """Return True if the path is inside excluded folders."""
    parts_lower = [part.lower() for part in path.parts]

    if ".venv" in parts_lower:
        return True

    # Exclude Data/processed subtree regardless of case.
    for i in range(len(parts_lower) - 1):
        if parts_lower[i] == "data" and parts_lower[i + 1] == "processed":
            return True

    return False


def ensure_destinations() -> None:
    """Create destination directories if needed."""
    T1_DEST.mkdir(parents=True, exist_ok=True)
    T2_DEST.mkdir(parents=True, exist_ok=True)
    IXI_ROOT.mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dst: Path) -> tuple[bool, bool]:
    """Move a file safely.

    Returns:
        (moved, skipped_existing)
    """
    if src.resolve() == dst.resolve():
        print(f"Already in place: {src.name}")
        return False, True

    if dst.exists():
        print(f"Already in place: {src.name}")
        return False, True

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True, False
    except Exception as exc:
        print(f"Error moving {src} -> {dst}: {exc}")
        return False, False


def find_matching_files(pattern: str) -> list[Path]:
    """Find files in project root recursively, excluding forbidden directories."""
    matches: list[Path] = []
    for path in PROJECT_ROOT.rglob(pattern):
        if not path.is_file():
            continue
        if is_excluded(path):
            continue
        matches.append(path)

    return sorted(matches, key=lambda p: str(p))


def move_modality_files(files: list[Path], dest_dir: Path, label: str) -> tuple[int, int]:
    """Move modality files and report progress.

    Returns:
        (moved_count, skipped_count)
    """
    moved_count = 0
    skipped_count = 0

    print(f"\n{label} files found: {len(files)}")
    for index, src in enumerate(files, start=1):
        dst = dest_dir / src.name
        moved, skipped = safe_move(src, dst)
        if moved:
            moved_count += 1
        if skipped:
            skipped_count += 1

        if index % PROGRESS_INTERVAL == 0:
            print(f"Processed {index}/{len(files)} {label} files...")

    print(f"{label} files moved: {moved_count}")
    return moved_count, skipped_count


def find_demographic_file() -> Path | None:
    """Find IXI demographic spreadsheet (.xls preferred over .xlsx)."""
    xls_files = find_matching_files("IXI.xls")
    xlsx_files = find_matching_files("IXI.xlsx")

    candidates = xls_files + xlsx_files
    if not candidates:
        return None

    # Prefer .xls to match requested destination naming.
    for candidate in candidates:
        if candidate.suffix.lower() == ".xls":
            return candidate
    return candidates[0]


def move_demographic_file() -> tuple[bool, str]:
    """Move demographic file to IXI root.

    Returns:
        (found, destination_filename)
    """
    source = find_demographic_file()
    if source is None:
        print("Demographic file not found (IXI.xls / IXI.xlsx)")
        return False, ""

    dest_name = "IXI.xls" if source.suffix.lower() == ".xls" else "IXI.xlsx"
    dest_path = IXI_ROOT / dest_name

    moved, skipped = safe_move(source, dest_path)
    if moved:
        print(f"Demographic file moved to: {dest_path}")
    elif skipped:
        print(f"Already in place: {dest_name}")
    else:
        print("Demographic file move failed")

    return True, dest_name


def print_structure_preview() -> None:
    """Print final IXI directory structure preview."""
    print("\nFinal directory structure preview:")
    print(f"{IXI_ROOT}")
    print("|- T1/")
    print("|- T2/")

    xls_path = IXI_ROOT / "IXI.xls"
    xlsx_path = IXI_ROOT / "IXI.xlsx"
    if xls_path.exists():
        print("|- IXI.xls")
    elif xlsx_path.exists():
        print("|- IXI.xlsx")
    else:
        print("|- IXI.xls (or IXI.xlsx) [not found]")

    try:
        t1_count = len([p for p in T1_DEST.glob("*.nii.gz") if p.is_file()])
        t2_count = len([p for p in T2_DEST.glob("*.nii.gz") if p.is_file()])
        print(f"T1 files currently in destination: {t1_count}")
        print(f"T2 files currently in destination: {t2_count}")
    except Exception as exc:
        print(f"Could not compute destination counts: {exc}")


def main() -> None:
    """Run IXI reorganization process."""
    print("=" * 80)
    print("IXI REORGANIZATION")
    print("=" * 80)

    ensure_destinations()

    t1_files = find_matching_files("IXI*-T1.nii.gz")
    t1_moved, _ = move_modality_files(t1_files, T1_DEST, "T1")

    t2_files = find_matching_files("IXI*-T2.nii.gz")
    t2_moved, _ = move_modality_files(t2_files, T2_DEST, "T2")

    demographic_found, _ = move_demographic_file()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total T1 files moved: {t1_moved}")
    print(f"Total T2 files moved: {t2_moved}")
    print(f"Demographic file found: {'yes' if demographic_found else 'no'}")

    print_structure_preview()


if __name__ == "__main__":
    main()
