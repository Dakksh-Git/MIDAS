from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "fig3_1_system_overview.png"


def _darken(hex_color: str, factor: float = 0.75) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    title: str,
    lines: list[str],
) -> None:
    shadow = FancyBboxPatch(
        (x + 0.006, y - 0.008),
        w,
        h,
        boxstyle="round,pad=0.02",
        facecolor="#CCCCCC",
        edgecolor="none",
        linewidth=0,
        alpha=0.5,
        zorder=1,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        facecolor=fill,
        edgecolor="black",
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h - 0.045,
        title,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=_darken(fill),
        fontfamily="Times New Roman",
        zorder=3,
    )

    start_y = y + h - 0.12
    step = 0.072
    for idx, line in enumerate(lines):
        ax.text(
            x + 0.012,
            start_y - idx * step,
            f"• {line}",
            ha="left",
            va="top",
            fontsize=9,
            fontfamily="Times New Roman",
            color="black",
            zorder=3,
        )


def main() -> None:
    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(20, 8), facecolor="white")
    fig.patch.set_facecolor("white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    width = 0.13
    height = 0.62
    y_center = 0.48
    y0 = y_center - height / 2
    x_positions = [0.01, 0.18, 0.35, 0.52, 0.69, 0.86]

    boxes = [
        (
            "Raw Datasets",
            "#AED6F1",
            [
                "BraTS2020 (369 patients)",
                "IXI (361 subjects)",
                "LUMIERE (599 timepoints)",
                "MS Dataset (60 patients)",
                "ReMIND (114 patients)",
            ],
        ),
        (
            "Preprocessing",
            "#A9DFBF",
            [
                "Load NIfTI / DICOM",
                "Resample to 128³",
                "Percentile clip + normalise",
                "Modality imputation",
                "Save as .npy (4,128,128,128)",
            ],
        ),
        (
            "Augmentation & Splits",
            "#FAD7A0",
            [
                "Augment Benign → 406",
                "Augment Inflammatory → 200",
                "Corruption validation",
                "70 / 15 / 15 stratified split",
                "Train: 1474 | Val: 316 | Test: 317",
            ],
        ),
        (
            "4-Branch 3D ResNet-18",
            "#D7BDE2",
            [
                "T1 Branch → 256-dim",
                "T1CE Branch → 256-dim",
                "T2 Branch → 256-dim",
                "FLAIR Branch → 256-dim",
                "Concatenate → 1024-dim",
            ],
        ),
        (
            "Training",
            "#FDFEBB",
            [
                "CrossEntropyLoss + class weights",
                "Adam lr = 0.0001",
                "CosineAnnealingLR",
                "Early stopping patience = 20",
                "Best epoch: 27 / 47",
            ],
        ),
        (
            "Outputs",
            "#FADBD8",
            [
                "Test Accuracy: 92.11%",
                "Macro AUC-ROC: 98.94%",
                "Confusion Matrix",
                "Integrated Gradients Maps",
                "Diagnostic GUI",
            ],
        ),
    ]

    for (title, fill, lines), x in zip(boxes, x_positions):
        _draw_box(ax, x, y0, width, height, fill, title, lines)

    for i in range(len(x_positions) - 1):
        start = (x_positions[i] + width + 0.004, y_center)
        end = (x_positions[i + 1] - 0.004, y_center)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 2.5},
            zorder=4,
        )

    fig.suptitle(
        "Fig. 3.1: MIDAS System Overview Block Diagram",
        fontsize=13,
        fontweight="bold",
        fontfamily="Times New Roman",
        y=0.98,
    )

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()