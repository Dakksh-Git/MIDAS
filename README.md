# MIDAS — Multi-modal Intelligent Diagnostic and Analysis System

> A 4-branch 3D deep learning system for brain MRI classification across 5 neurological conditions, trained on 1,700+ patient scans with 92.3% test accuracy and 99.1% macro AUC-ROC.

---

## Architecture

MIDAS uses a **4-branch 3D ResNet-18** (~33.7M parameters), with one dedicated branch per MRI modality:

| Branch | Modality | Role |
|--------|----------|------|
| Branch 1 | T1 | Structural baseline |
| Branch 2 | T1CE | Contrast-enhanced lesion detection |
| Branch 3 | T2 | Edema and fluid regions |
| Branch 4 | FLAIR | White matter lesions |

Each branch independently encodes its modality through residual 3D convolutions. Branch outputs are concatenated and passed through a shared classifier head for 5-class prediction. This design preserves modality-specific features while enabling cross-modal fusion.

```
T1   ──► ResNet-18 Branch ──┐
T1CE ──► ResNet-18 Branch ──┤
                             ├──► Concat ──► FC ──► 5-class output
T2   ──► ResNet-18 Branch ──┤
FLAIR──► ResNet-18 Branch ──┘
```

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **92.3%** |
| Macro AUC-ROC | **99.1%** |
| Parameters | ~33.7M |
| Test Set Size | 256 scans |

Training split: **1192 train / 255 val / 256 test** across a 1,700+ patient hybrid dataset.

---

## Dataset

MIDAS is trained on a hybrid dataset combining three public sources — MRI-only, no CT or PET data used:

| Dataset | Modalities | Contribution |
|---------|------------|--------------|
| [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/) | T1, T1CE, T2, FLAIR | High-grade & low-grade glioma |
| [ReMIND](https://www.cancerimagingarchive.net/collection/remind/) | T1, T1CE, T2, FLAIR | Clinical glioma with post-surgical scans |
| [IXI](https://brain-development.org/ixi-dataset/) | T1, T2 | Healthy baseline controls |

**5 output classes:** Healthy, High-Grade Glioma, Low-Grade Glioma, Meningioma, Other Abnormality

### Preprocessing Pipeline
- Skull stripping and bias field correction via SimpleITK
- Registration to MNI152 space
- Normalization to zero mean, unit variance per volume
- Resized to fixed volumetric shape for batch consistency
- Class-balanced augmentation (random flips, rotations, intensity jitter)

---

## Explainability

MIDAS uses **Integrated Gradients (XAI)** to generate per-voxel attribution maps, highlighting which regions of each MRI modality most influenced the model's prediction.

Grad-CAM++ and Guided Backpropagation were evaluated and rejected — IG was chosen for its theoretical soundness (completeness axiom) and stability across modalities.

Attribution maps are exported per-modality as `.nii.gz` volumes and rendered in the GUI.

---

## Project Structure

```
MIDAS/
├── src/                        # Core source code
│   ├── model.py                # 4-branch 3D ResNet-18 architecture
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation & metrics
│   ├── preprocess.py           # Data preprocessing pipeline
│   ├── rebuild_splits.py       # Train/val/test split generation
│   ├── augment.py              # Class balancing & augmentation
│   ├── gradcam.py              # Integrated Gradients explainability
│   ├── gui.py                  # Desktop inference GUI
│   ├── explore_datasets.py     # Dataset exploration utility
│   ├── check_remind_mapping.py # ReMIND label validation
│   └── plots/                  # Visualization scripts
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
│   ├── Raw/                    # Source datasets (not tracked)
│   ├── processed/MRI/          # Preprocessed .npy volumes (not tracked)
│   ├── splits/                 # train.csv / val.csv / test.csv
│   └── metadata/               # Dataset metadata
│
├── checkpoints/                # Model weights (not tracked)
├── outputs/                    # Logs, IG maps, classification report
├── restructure_project.py      # Directory migration script (May 2026)
└── README.md
```

---

## Setup

### Requirements
```bash
pip install torch torchvision numpy nibabel SimpleITK scikit-learn captum
```

> Tested on Python 3.10+, PyTorch 2.x, CUDA 12.x (RTX 4060 8GB)

### Inference (GUI)
```bash
python src/gui.py
```

### Training
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py
```

---

## Tech Stack

`Python` `PyTorch` `NumPy` `NiBabel` `SimpleITK` `Scikit-learn` `Captum` `CUDA`

---

## License

See [LICENSE](LICENSE) for details.

---

*B.Tech Minor-II Project — Electronics & Communication Engineering (EC-ACT), JIIT 2023–2027*
