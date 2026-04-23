"""Diagnostic utility for ReMIND clinical-to-class mapping.

This script reuses mapping helpers from preprocess.py so the behavior remains
consistent with the preprocessing pipeline.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from preprocess import (
    CLASS_NAMES,
    REMIND_CLINICAL,
    format_remind_case_id,
    map_remind_label,
    normalize_text,
    parse_grade,
)


def reason_for_mapping(histopathology: object, grade_value: object, label: int | None) -> str:
    hist = normalize_text(histopathology)
    hist_lower = hist.lower()
    grade = parse_grade(grade_value)

    if label is None:
        if not hist_lower:
            return "Skipped: empty/unrecognized histopathology"
        return "Skipped: unrecognized histopathology"

    if hist_lower == "astrocytoma" and grade not in {"1", "2", "3", "4"}:
        return "Defaulted: astrocytoma with missing/non-standard WHO grade -> class 1"

    return "Mapped successfully"


def main() -> None:
    clinical_path = Path(REMIND_CLINICAL)
    if not clinical_path.exists():
        raise FileNotFoundError(f"ReMIND clinical file not found: {clinical_path}")

    df = pd.read_excel(clinical_path)
    required_columns = {"Case Number", "Histopathology", "WHO Grade"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Clinical file is missing required columns: {sorted(missing)}")

    class_counts: Counter[int] = Counter()
    unrecognized_histopathologies = set()
    defaulted_histopathologies = set()

    print("=" * 100)
    print("ReMIND Mapping Diagnostics")
    print("=" * 100)

    for _, row in df.iterrows():
        case_id = format_remind_case_id(row["Case Number"])
        hist = normalize_text(row["Histopathology"])
        who_grade = normalize_text(row["WHO Grade"])
        label = map_remind_label(hist, who_grade)
        reason = reason_for_mapping(hist, who_grade, label)

        if label is not None:
            class_counts[label] += 1

        if reason.startswith("Skipped"):
            unrecognized_histopathologies.add(hist if hist else "<EMPTY>")
        if reason.startswith("Defaulted"):
            defaulted_histopathologies.add(hist if hist else "<EMPTY>")

        print(f"Case Number: {case_id}")
        print(f"  Histopathology: {hist if hist else '<EMPTY>'}")
        print(f"  WHO Grade: {who_grade if who_grade else '<EMPTY>'}")
        print(f"  Mapped class label: {label if label is not None else 'SKIPPED'}")
        print(f"  Reason: {reason}")

    print("\n" + "=" * 100)
    print("Class Counts")
    print("=" * 100)
    for label in sorted(CLASS_NAMES):
        print(f"Class {label} ({CLASS_NAMES[label]}): {class_counts.get(label, 0)}")

    print("\n" + "=" * 100)
    print("Unrecognized Histopathology Values")
    print("=" * 100)
    if not unrecognized_histopathologies:
        print("None")
    else:
        for value in sorted(unrecognized_histopathologies):
            print(value)

    print("\n" + "=" * 100)
    print("Defaulted Histopathology Values")
    print("=" * 100)
    if not defaulted_histopathologies:
        print("None")
    else:
        for value in sorted(defaulted_histopathologies):
            print(value)


if __name__ == "__main__":
    main()
