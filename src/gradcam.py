from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, gaussian_filter
from skimage.filters import threshold_otsu
from tqdm import tqdm

from model import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"
TEST_CSV = PROJECT_ROOT / "Data" / "splits" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = {0: "Malignant", 1: "Benign", 2: "Normal", 3: "Scar", 4: "Inflammatory"}
BRANCH_NAMES = ["T1", "T1CE", "T2", "FLAIR"]
NUM_CLASSES = 5
IG_STEPS = 50


class IntegratedGradients:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def compute(self, input_tensor: torch.Tensor, target_class: int, steps: int = IG_STEPS) -> np.ndarray:
        baseline = torch.zeros_like(input_tensor)
        scaled_inputs = [baseline + (float(step) / steps) * (input_tensor - baseline) for step in range(steps + 1)]

        gradients: list[torch.Tensor] = []
        batch_size = 5

        for start in range(0, len(scaled_inputs), batch_size):
            batch = torch.cat(scaled_inputs[start : start + batch_size], dim=0).requires_grad_(True)
            self.model.zero_grad(set_to_none=True)

            outputs = self.model(batch)
            scores = outputs[:, target_class].sum()
            scores.backward()

            gradients.append(batch.grad.detach().clone())

        all_gradients = torch.cat(gradients, dim=0)
        avg_grads = all_gradients.mean(dim=0)
        integrated_grads = (input_tensor - baseline) * avg_grads

        saliency = torch.sum(torch.abs(integrated_grads[0]), dim=0).detach().cpu().numpy()
        p99 = np.percentile(saliency, 99)
        if p99 > 0:
            saliency = np.clip(saliency / p99, 0, 1)

        return saliency.astype(np.float32)


def get_brain_mask(volume: np.ndarray) -> np.ndarray:
    return volume.mean(axis=0) > 0.01


def find_peak_slices(saliency: np.ndarray, margin: int = 15) -> tuple[int, int, int]:
    margin = 5
    threshold = np.percentile(saliency, 98)
    hot = saliency >= threshold
    coords = np.argwhere(hot)
    if len(coords) == 0:
        mid = saliency.shape[0] // 2
        return mid, mid, mid

    axial = int(np.clip(np.median(coords[:,0]), margin, saliency.shape[0]-margin))
    coronal = int(np.clip(np.median(coords[:,1]), margin, saliency.shape[1]-margin))
    sagittal = int(np.clip(np.median(coords[:,2]), margin, saliency.shape[2]-margin))
    return axial, coronal, sagittal


def visualize_sample(
    volume: np.ndarray,
    saliency: np.ndarray,
    true_label: str,
    pred_label: str,
    pred_prob: float,
    subject_id: str,
    output_dir: Path,
) -> Path:
    label = next((key for key, value in CLASS_NAMES.items() if value == true_label), -1)
    brain_mask = get_brain_mask(volume)
    saliency = saliency * brain_mask.astype(np.float32)

    brain_vals = saliency[brain_mask]
    if brain_vals.size > 0:
        p99 = np.percentile(brain_vals, 99)
        if p99 > 0:
            saliency = np.clip(saliency / p99, 0, 1)

    axial, coronal, sagittal = find_peak_slices(saliency, margin=15)

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle(
        f"True: {true_label} | Pred: {pred_label} ({pred_prob:.1%})\nIntegrated Gradients",
        fontsize=13,
    )

    col_titles = [f"Axial (z={axial})", f"Coronal (y={coronal})", f"Sagittal (x={sagittal})"]
    im = None

    for row in range(4):
        mri = volume[row]
        for col, sl in enumerate([axial, coronal, sagittal]):
            ax = axes[row, col]
            if col == 0:
                mri_slice = mri[sl, :, :]
                sal_slice = saliency[sl, :, :]
            elif col == 1:
                mri_slice = mri[:, sl, :]
                sal_slice = saliency[:, sl, :]
            else:
                mri_slice = mri[:, :, sl]
                sal_slice = saliency[:, :, sl]

            try:
                thresh_val = threshold_otsu(mri_slice)
            except Exception:
                thresh_val = 0.05
            brain_mask_2d = (mri_slice > thresh_val).astype(np.float32)
            from scipy.ndimage import binary_fill_holes

            brain_mask_2d = binary_fill_holes(brain_mask_2d).astype(np.float32)
            brain_mask_2d = binary_erosion(brain_mask_2d, iterations=3).astype(np.float32)
            saliency_2d = sal_slice * brain_mask_2d
            saliency_smooth = gaussian_filter(saliency_2d, sigma=2.5)

            positive_vals = saliency_smooth[saliency_smooth > 0]
            if positive_vals.size > 0:
                thresh = float(np.percentile(positive_vals, 85))
            else:
                thresh = 0.0

            ax.imshow(mri_slice.T, cmap="gray", origin="lower")
            saliency_smooth[saliency_smooth < thresh] = 0
            cmap = "RdGy_r" if label == 2 else "hot"
            alpha = 0.45 if label == 2 else 0.6
            im = ax.imshow(saliency_smooth.T, cmap=cmap, alpha=alpha, origin="lower", vmin=thresh)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(BRANCH_NAMES[row], fontsize=9)
            ax.axis("off")

    cax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    sm = cm.ScalarMappable(cmap="hot", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Attribution")
    fig.subplots_adjust(bottom=0.08)
    out_path = Path(output_dir) / f"{subject_id}_ig.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_model() -> torch.nn.Module:
    model = get_model(device=DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def _select_samples(model: torch.nn.Module, test_df: pd.DataFrame) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running inference", unit="sample"):
        filepath = Path(str(row["filepath"]))
        if not filepath.exists():
            continue

        volume = np.load(filepath).astype(np.float32)
        input_tensor = torch.from_numpy(volume).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            pred_label = int(torch.argmax(logits, dim=1).item())
            pred_prob = float(probabilities[0, pred_label].item())

        predictions.append(
            {
                "filepath": filepath,
                "true_label": int(row["label"]),
                "pred_label": pred_label,
                "pred_prob": pred_prob,
                "is_augmented": bool(row.get("is_augmented", False)),
                "source": str(row.get("source", "")).lower() if not pd.isna(row.get("source", "")) else "",
            }
        )

    selected_by_class: dict[int, list[dict[str, object]]] = {cls: [] for cls in range(NUM_CLASSES)}
    for sample in predictions:
        true_label = int(sample["true_label"])
        pred_label = int(sample["pred_label"])
        if true_label == pred_label:
            selected_by_class[true_label].append(sample)

    rng = np.random.default_rng(42)
    selected_samples: list[dict[str, object]] = []
    for cls in range(NUM_CLASSES):
        pool = selected_by_class[cls]
        if cls == 1:
            non_aug_brats = [
                item
                for item in pool
                if (not bool(item["is_augmented"])) and str(item.get("source", "")) == "brats"
            ]
            if len(non_aug_brats) > 0:
                selected = non_aug_brats
            else:
                augmented = [item for item in pool if bool(item["is_augmented"])]
                selected = augmented if len(augmented) > 0 else pool
        elif cls == 2:
            excluded_subject_ids = {"IXI535", "IXI574", "IXI305", "IXI028"}
            selected = [
                item
                for item in pool
                if (not bool(item["is_augmented"]))
                and str(item.get("source", "")) == "ixi"
                and "IXI" in str(item["filepath"])
                and not any(subject_id in str(item["filepath"]) for subject_id in excluded_subject_ids)
            ]
            rng.shuffle(selected)
        else:
            non_aug = [item for item in pool if not bool(item["is_augmented"])]
            selected = non_aug if len(non_aug) > 0 else pool
        if selected:
            selected_samples.append(selected[0] if cls == 2 else selected[int(rng.integers(len(selected)))])

    return selected_samples


def main() -> None:
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_model()
    test_df = pd.read_csv(TEST_CSV)
    missing = {"filepath", "label"} - set(test_df.columns)
    if missing:
        raise ValueError(f"test.csv is missing columns: {sorted(missing)}")

    selected_samples = _select_samples(model, test_df)
    if not selected_samples:
        print("No correctly predicted samples were found for Integrated Gradients generation.")
        return

    ig = IntegratedGradients(model)
    saved_paths: list[Path] = []

    for sample in tqdm(selected_samples, desc="Generating Integrated Gradients", unit="sample"):
        filepath = Path(str(sample["filepath"]))
        volume = np.load(filepath).astype(np.float32)
        input_tensor = torch.from_numpy(volume).unsqueeze(0).to(DEVICE)

        saliency = ig.compute(input_tensor, int(sample["pred_label"]), steps=IG_STEPS)
        out_path = visualize_sample(
            volume=volume,
            saliency=saliency,
            true_label=CLASS_NAMES[int(sample["true_label"])],
            pred_label=CLASS_NAMES[int(sample["pred_label"])],
            pred_prob=float(sample["pred_prob"]),
            subject_id=filepath.stem,
            output_dir=OUTPUT_DIR,
        )
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")

    print(f"\nTotal saved files: {len(saved_paths)}")


if __name__ == "__main__":
    main()
