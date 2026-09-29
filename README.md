# MIDAS — Multi-modal Intelligent Diagnostic and Analysis System

> A 4-branch 3D deep learning system for five-class brain MRI classification, trained on patient-level, class-stratified splits: 91.03% test accuracy, 74.84% macro F1, and 98.06% macro AUC-ROC on 223 patient-disjoint test volumes.

---

## Architecture

MIDAS uses a **4-branch, half-width 3D ResNet-18** (~33.7M parameters), with one dedicated branch per MRI sequence:

| Branch | Sequence | Role |
|--------|----------|------|
| Branch 1 | T1 | Structural baseline |
| Branch 2 | T1CE | Contrast-enhanced lesion detection |
| Branch 3 | T2 | Edema and fluid regions |
| Branch 4 | FLAIR | White matter lesions |

Each branch independently encodes its sequence through residual 3D convolutions with half-width filters (to fit an 8 GB consumer GPU), producing a 256-dimensional sequence-specific representation. The four vectors are concatenated and passed through a shared classification head (512-unit projection, batch normalisation, ReLU, dropout at rate 0.5) for 5-class prediction.

```
T1   ──► ResNet-18 Branch ──┐
T1CE ──► ResNet-18 Branch ──┤
                             ├──► Concat ──► FC ──► 5-class output
T2   ──► ResNet-18 Branch ──┤
FLAIR──► ResNet-18 Branch ──┘
```

---

## Results

Evaluation uses a **patient-level, class-stratified split** (no patient's volumes appear in more than one of train/val/test; augmentation is applied only to the training fold's own volumes, after splitting).

| Class | N (test) | Accuracy | F1 |
|-------|----------|----------|-----|
| Malignant | 55 | 96.36% | 84.80% |
| Benign | 16 | 0.00% | 0.00% |
| Normal | 55 | 100.00% | 100.00% |
| Scar | 87 | 98.85% | 99.43% |
| Inflammatory | 10 | 90.00% | 90.00% |
| **Overall** | **223** | **91.03%** | Macro F1: **74.84%** |

Macro AUC-ROC (one-vs-rest): **98.06%**.

**Known limitation — Malignant-vs-Benign discrimination.** The current model classifies 0/16 Benign test cases correctly (all predicted Malignant). Per-class AUC analysis shows this is not a leakage artefact or simple decision-threshold problem: Benign's own one-vs-rest AUC (93.66%) is inflated by easy tumour/non-tumour separation, and the isolated Malignant-vs-Benign head-to-head AUC is a moderate 76.93%, with most misclassified cases confidently (not borderline) assigned to Malignant. This reflects a genuine separability limitation in the current trained model. See the paper for the full diagnostic trail, including two independent data-provenance defects (a ReMIND single-sequence-duplication artefact and a histopathology-grade labelling bug) found and fixed during the investigation, and their quantified before/after impact.

Training split (post-augmentation): **1,360 train / 214 val / 223 test**, drawn from **994 unique patients** across 5 sources.

---

## Dataset

MIDAS combines five open-access sources — MRI-only, no CT or PET data used:

| Dataset | Sequences | Contribution |
|---------|-----------|--------------|
| [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/) | T1, T1CE, T2, FLAIR (genuine) | Malignant (HGG) & Benign (LGG) |
| [LUMIERE](https://www.nature.com/articles/s41597-022-01881-7) | T1, T1CE, T2, FLAIR (genuine) | Scar (post-treatment necrosis), 91 patients × longitudinal timepoints |
| [IXI](https://brain-development.org/ixi-dataset/) | T1, T2 (T1CE/FLAIR duplicated — disclosed limitation) | Normal (healthy controls) |
| [ReMIND](https://www.cancerimagingarchive.net/collection/remind/) | T1CE/T2/FLAIR extracted by DICOM SeriesDescription per patient; T1 duplicated where unavailable | Multi-class (histopathology + WHO-grade based) |
| MS imaging cohort | T1, T2, FLAIR (T1CE duplicated — disclosed limitation) | Inflammatory (demyelinating disease) |

**5 output classes:** Malignant, Benign, Normal, Scar, Inflammatory

### Preprocessing Pipeline
- NIfTI via NiBabel; DICOM via PyDicom + SimpleITK for slice ordering
- Trilinear resampling to a fixed 128×128×128 volumetric shape
- Per-channel intensity clipped to the 1st–99th percentile, then rescaled to [0, 1]
- Patient-level, class-stratified group-shuffle split (seed 42)
- Class-balanced augmentation (rotation ±15°, reflection, Gaussian noise, intensity jitter) — applied only to the training fold's own volumes, after splitting

---

## Explainability

MIDAS uses **Integrated Gradients (XAI)** to generate per-voxel attribution maps, highlighting which regions of each MRI sequence most influenced the model's prediction.

Grad-CAM and Guided Backpropagation were evaluated and rejected — IG was chosen for its theoretical soundness (completeness/sensitivity axioms) and input-resolution attribution.

Attribution centroid selection uses the top 2% of attribution voxels to pick the displayed axial/coronal/sagittal planes.

---

## Project Structure

```
MIDAS/
├── src/                                # Core source code
│   ├── model.py                        # 4-branch 3D ResNet-18 architecture
│   ├── train.py                        # Training loop (AMP, batch size 8)
│   ├── evaluate.py                     # Evaluation & metrics
│   ├── preprocess.py                   # Data preprocessing pipeline
│   ├── rebuild_splits.py               # Patient-level split generation
│   ├── rebuild_splits_filelevel_OLD.py # Superseded file-level split (kept for audit)
│   ├── augment_train.py                # Train-fold-only augmentation, post-split
│   ├── augment.py                      # Legacy augmentation (pre-fix pipeline)
│   ├── gradcam.py                      # Integrated Gradients explainability
│   ├── gui.py                          # Desktop inference GUI
│   ├── explore_datasets.py             # Dataset exploration utility
│   ├── check_remind_mapping.py         # ReMIND label validation
│   └── plots/                          # Visualization scripts
│       ├── plot_preprocessing_flowchart.py
│       ├── plot_system_overview.py
│       └── plot_training_curves.py
│
├── scripts/                    # Setup & data utilities
│   ├── kaggle_setup.py
│   ├── download_scar.py
│   ├── reorganize.py
│   ├── reorganize_ixi.py
│   └── remind/
│
├── Data/
│   ├── Raw/                            # Source datasets (not tracked)
│   ├── processed/MRI/                  # Preprocessed .npy volumes (not tracked)
│   ├── splits/                         # train.csv / val.csv / test.csv (patient-level)
│   │   └── backup_pre_patient_split/   # Original file-level splits, kept for audit
│   └── metadata/                       # Dataset metadata
│
├── checkpoints/                 # Model weights (not tracked)
├── outputs/                     # Logs, IG maps, classification report
│   └── *_archive/                # Snapshots from each stage of the leakage/audit fix
├── restructure_project.py       # Directory migration script (May 2026)
└── README.md
```

---

## Setup

### Requirements
```bash
pip install torch torchvision numpy nibabel SimpleITK scikit-learn captum pydicom
```

> Tested on Python 3.10+, PyTorch 2.6.0 (CUDA 12.4), RTX 4060 8GB

### Data pipeline (in order)
```bash
python src/preprocess.py          # Raw datasets -> normalized .npy volumes
python src/rebuild_splits.py      # Patient-level, class-stratified train/val/test split
python src/augment_train.py       # Train-fold-only augmentation, post-split
```

### Training
```bash
python src/train.py               # add --fresh to ignore existing checkpoints
```

### Evaluation
```bash
python src/evaluate.py
python src/gradcam.py             # Integrated Gradients attribution maps
```

### Inference (GUI)
```bash
python src/gui.py
```

---

## Methodology notes

This project underwent a retrospective data-provenance audit after an initial file-level evaluation reported higher (but leakage-inflated) headline numbers. Three independent issues were found and fixed:

1. **Patient-level leakage**: the original split stratified individual files (not patients) and applied augmentation before splitting, letting a patient's own volumes — and augmented derivatives of a volume — cross train/val/test boundaries.
2. **ReMIND channel-duplication defect**: DICOM preprocessing collapsed each patient to their single largest series, duplicated across all 4 input channels, inconsistently per patient (sometimes T1CE, sometimes T2). Fixed by matching each series to its clinical role via `SeriesDescription`.
3. **Oligodendroglioma grade-labelling bug**: WHO grade 3 (anaplastic/high-grade) oligodendroglioma cases were being labelled Benign unconditionally, inconsistent with the grade-aware handling already applied to astrocytoma.

The numbers above reflect the corrected pipeline. See the accompanying paper for the full before/after comparison and diagnostic analysis.

---

## Tech Stack

`Python` `PyTorch` `NumPy` `NiBabel` `PyDicom` `SimpleITK` `Scikit-learn` `Captum` `CUDA`

---

## License

See [LICENSE](LICENSE) for details.

---

*B.Tech Minor-II Project — Electronics & Communication Engineering (EC-ACT), JIIT 2023–2027*
