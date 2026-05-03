from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "fig3_2_preprocessing.png"


def draw_rect_node(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    title: str,
    content: str,
) -> None:
    x0 = x - width / 2
    y0 = y - height / 2

    shadow = FancyBboxPatch(
        (x0 + 0.008, y0 - 0.008),
        width,
        height,
        boxstyle="round,pad=0.015",
        facecolor="#CCCCCC",
        edgecolor="none",
        alpha=0.5,
        zorder=1,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.015",
        facecolor=fill,
        edgecolor="black",
        linewidth=1.6,
        zorder=2,
    )
    ax.add_patch(box)

    ax.text(
        x,
        y + 0.02,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        fontfamily="Times New Roman",
        zorder=3,
    )
    ax.text(
        x,
        y - 0.015,
        content,
        ha="center",
        va="center",
        fontsize=9,
        fontfamily="Times New Roman",
        zorder=3,
    )


def draw_side_rect_node(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    text: str,
) -> None:
    x0 = x - width / 2
    y0 = y - height / 2

    shadow = FancyBboxPatch(
        (x0 + 0.008, y0 - 0.008),
        width,
        height,
        boxstyle="round,pad=0.015",
        facecolor="#CCCCCC",
        edgecolor="none",
        alpha=0.5,
        zorder=1,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.015",
        facecolor=fill,
        edgecolor="black",
        linewidth=1.6,
        zorder=2,
    )
    ax.add_patch(box)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        fontfamily="Times New Roman",
        zorder=3,
    )


def draw_diamond_node(ax: plt.Axes, x: float, y: float, width: float, height: float, fill: str, text: str) -> None:
    points_shadow = [
        (x, y + height / 2),
        (x + width / 2, y),
        (x, y - height / 2),
        (x - width / 2, y),
    ]
    points_shadow = [(px + 0.008, py - 0.008) for px, py in points_shadow]
    shadow = Polygon(points_shadow, closed=True, facecolor="#CCCCCC", edgecolor="none", alpha=0.5, zorder=1)
    ax.add_patch(shadow)

    points = [
        (x, y + height / 2),
        (x + width / 2, y),
        (x, y - height / 2),
        (x - width / 2, y),
    ]
    diamond = Polygon(points, closed=True, facecolor=fill, edgecolor="black", linewidth=1.6, zorder=2)
    ax.add_patch(diamond)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        fontfamily="Times New Roman",
        zorder=3,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
        zorder=4,
    )


def draw_arrow_label(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=9,
        fontstyle="italic",
        fontfamily="Times New Roman",
        ha="center",
        va="center",
        zorder=5,
    )


def main() -> None:
    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(10, 18), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    main_x = 0.5
    rect_w, rect_h = 0.52, 0.09
    diamond_w, diamond_h = 0.38, 0.07
    side_w, side_h = 0.28, 0.08
    side_x = 0.76

    y1, y2, y3, y4, y5, y6, y7, y8 = 0.93, 0.80, 0.67, 0.54, 0.42, 0.29, 0.17, 0.06

    draw_diamond_node(ax, main_x, y1, diamond_w, diamond_h, "#FDFEBB", "Start\nRaw MRI Volume\n(NIfTI / DICOM)")

    draw_rect_node(
        ax,
        main_x,
        y2,
        rect_w,
        rect_h,
        "#AED6F1",
        "Step 1: Load Volume",
        "NiBabel for NIfTI\nPyDicom + SimpleITK for DICOM\nAuto series sorting",
    )

    draw_rect_node(
        ax,
        main_x,
        y3,
        rect_w,
        rect_h,
        "#AED6F1",
        "Step 2: Spatial Resampling",
        "SciPy zoom - trilinear interpolation\nTarget shape: (128, 128, 128)\n~8.4 MB per volume at float32",
    )

    draw_rect_node(
        ax,
        main_x,
        y4,
        rect_w,
        rect_h,
        "#AED6F1",
        "Step 3: Intensity Normalisation",
        "Clip to 1st-99th percentile per channel\nMin-max scale to [0, 1]\nApplied independently per channel",
    )

    draw_diamond_node(ax, main_x, y5, diamond_w, diamond_h, "#FAD7A0", "Missing T1CE?\n(IXI or MS dataset)")

    draw_side_rect_node(ax, side_x, y5, side_w, side_h, "#FADBD8", "Duplicate T1 -> T1CE\nIXI: also T2 -> FLAIR")

    draw_rect_node(
        ax,
        main_x,
        y6,
        rect_w,
        rect_h,
        "#A9DFBF",
        "Step 4: Stack 4 Channels",
        "[T1, T1CE, T2, FLAIR]\nShape: (4, 128, 128, 128) float32",
    )

    draw_diamond_node(ax, main_x, y7, diamond_w, diamond_h, "#FAD7A0", "Augmented file?\nRun corruption check")

    draw_side_rect_node(ax, side_x, y7, side_w, side_h, "#FADBD8", "Delete file\nstd < 0.01 or\nmean not in [0.05, 0.95]")

    draw_rect_node(
        ax,
        main_x,
        y8,
        rect_w,
        rect_h,
        "#A9DFBF",
        "Step 5: Save",
        "{source}_{id}_{label}.npy\nFinal shape: (4, 128, 128, 128)",
    )

    draw_arrow(ax, (main_x, y1), (main_x, y2))
    draw_arrow(ax, (main_x, y2), (main_x, y3))
    draw_arrow(ax, (main_x, y3), (main_x, y4))
    draw_arrow(ax, (main_x, y4), (main_x, y5))

    draw_arrow(ax, (main_x, y5), (side_x, y5))
    draw_arrow_label(ax, 0.64, y5 + 0.02, "YES")

    draw_arrow(ax, (side_x, y5), (main_x, y6))

    draw_arrow(ax, (main_x, y5), (main_x, y6))
    draw_arrow_label(ax, 0.53, (y5 + y6) / 2, "NO")

    draw_arrow(ax, (main_x, y6), (main_x, y7))

    draw_arrow(ax, (main_x, y7), (side_x, y7))
    draw_arrow_label(ax, 0.64, y7 + 0.02, "FAIL")

    draw_arrow(ax, (main_x, y7), (main_x, y8))
    draw_arrow_label(ax, 0.53, (y7 + y8) / 2, "PASS")

    fig.suptitle(
        "Fig. 3.2: Preprocessing Pipeline Flowchart",
        fontsize=13,
        fontweight="bold",
        fontfamily="Times New Roman",
        y=0.99,
    )

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()