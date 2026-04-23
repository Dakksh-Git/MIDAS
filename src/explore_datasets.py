"""
MIDAS: Multi-modal Intelligent Diagnostic and Analysis System
Dataset Exploration Script

Explores three datasets:
1. BraTS2020 (MRI with segmentation)
2. ReMIND (MRI with clinical data)
3. CAI2R (PET + MRI combined multimodal)
"""

import os
import glob
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import nibabel as nib
import pandas as pd
import h5py
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================================
# DATASET 1: BraTS2020 (MRI)
# ============================================================================

def explore_brats2020():
    print("\n" + "="*80)
    print("DATASET 1: BraTS2020 (MRI with Segmentation)")
    print("="*80)
    
    try:
        brats_path = str(PROJECT_ROOT / "Data" / "Raw" / "BraTS2020" / "training")
        
        # Find all patient folders
        patient_folders = sorted([d for d in os.listdir(brats_path) 
                                 if d.startswith("BraTS20_Training_")])
        
        print(f"\nTotal patient folders found: {len(patient_folders)}")
        
        # Track segmentation labels
        label_combinations = defaultdict(int)
        
        for i, patient in enumerate(patient_folders[:5]):  # Show first 5 in detail
            patient_dir = os.path.join(brats_path, patient)
            
            # Load T1CE image
            t1ce_file = os.path.join(patient_dir, f"{patient}_t1ce.nii")
            if os.path.exists(t1ce_file):
                mri_img = nib.load(t1ce_file)
                mri_data = mri_img.get_fdata()
                print(f"\n[{patient}]")
                print(f"  T1CE: shape={mri_data.shape}, dtype={mri_data.dtype}, "
                      f"min={mri_data.min():.4f}, max={mri_data.max():.4f}")
            
            # Load segmentation
            seg_file = os.path.join(patient_dir, f"{patient}_seg.nii")
            if os.path.exists(seg_file):
                seg_img = nib.load(seg_file)
                seg_data = seg_img.get_fdata()
                unique_labels = sorted(np.unique(seg_data).astype(int))
                print(f"  Segmentation: unique labels = {unique_labels}")
                
                # Track label combinations for all patients
                label_tuple = tuple(unique_labels)
                label_combinations[label_tuple] += 1
        
        # Quick sample of label statistics (every 10th patient for speed)
        print(f"\n--- Sampling label statistics (every 10th patient) ---")
        for patient in patient_folders[::10]:
            patient_dir = os.path.join(brats_path, patient)
            seg_file = os.path.join(patient_dir, f"{patient}_seg.nii")
            if os.path.exists(seg_file):
                seg_img = nib.load(seg_file)
                seg_data = seg_img.get_fdata()
                unique_labels = tuple(sorted(np.unique(seg_data).astype(int)))
                label_combinations[unique_labels] += 1
        
        print(f"\nLabel Combinations Found (from sample):")
        for labels, count in sorted(label_combinations.items()):
            print(f"  {labels}: {count} samples")
        
        print(f"\n✓ BraTS2020 Dataset Summary:")
        print(f"  Total patients: {len(patient_folders)}")
        print(f"  Image shape: (240, 240, 155)")
        print(f"  Available modalities: t1, t1ce, t2, flair, seg (segmentation)")
        
    except Exception as e:
        print(f"\n✗ Error exploring BraTS2020: {type(e).__name__}: {e}")


# ============================================================================
# DATASET 2: ReMIND (MRI + Clinical Data)
# ============================================================================

def explore_remind():
    print("\n" + "="*80)
    print("DATASET 2: ReMIND (MRI + Clinical Data)")
    print("="*80)
    
    try:
        # Load clinical data
        clinical_file = str(PROJECT_ROOT / "Data" / "Raw" / "ReMIND" / "clinical_data.xlsx")
        df = pd.read_excel(clinical_file)
        
        print(f"\n1. Clinical Data Overview:")
        print(f"   Total patients: {len(df)}")
        print(f"   Columns loaded: {len(df.columns)}")
        
        # Histopathology analysis
        if "Histopathology" in df.columns:
            print(f"\n2. Histopathology (Diagnosis/Tumor Type):")
            print(f"   Unique values: {df['Histopathology'].nunique()}")
            print(f"   Values:")
            for val, count in df['Histopathology'].value_counts().items():
                print(f"     - {val}: {count}")
        
        # WHO Grade analysis
        if "WHO Grade" in df.columns:
            print(f"\n3. WHO Grade:")
            print(f"   Unique values: {df['WHO Grade'].nunique()}")
            print(f"   Values:")
            for val, count in df['WHO Grade'].value_counts().items():
                print(f"     - {val}: {count}")
        
        # DICOM file exploration
        print(f"\n4. DICOM Files:")
        images_path = str(PROJECT_ROOT / "Data" / "Raw" / "ReMIND" / "images")
        
        # Check ReMIND-001
        remind_001_path = os.path.join(images_path, "ReMIND-001")
        if os.path.exists(remind_001_path):
            dcm_files = glob.glob(os.path.join(remind_001_path, "**/*.dcm"), 
                                 recursive=True)
            print(f"   ReMIND-001 total .dcm files: {len(dcm_files)}")
            
            if dcm_files:
                # Load and inspect first DICOM
                first_dcm = dcm_files[0]
                try:
                    ds = pydicom.dcmread(first_dcm)
                    print(f"\n   First DICOM file: {os.path.basename(first_dcm)}")
                    print(f"     Modality: {ds.Modality if 'Modality' in ds else 'N/A'}")
                    print(f"     Pixel array shape: {ds.pixel_array.shape}")
                    print(f"     Pixel array dtype: {ds.pixel_array.dtype}")
                    
                    if 'SliceThickness' in ds:
                        print(f"     Slice Thickness: {ds.SliceThickness}")
                    if 'PixelSpacing' in ds:
                        print(f"     Pixel Spacing: {ds.PixelSpacing}")
                except Exception as e:
                    print(f"     Error reading DICOM: {e}")
        
        # Count all DICOM files across all patients
        all_dcm_files = glob.glob(os.path.join(images_path, "**/*.dcm"), 
                                 recursive=True)
        print(f"\n   Total .dcm files across all patients: {len(all_dcm_files)}")
        
        print(f"\n✓ ReMIND Dataset Summary:")
        print(f"  Clinical records: {len(df)}")
        print(f"  Total DICOM files: {len(all_dcm_files)}")
        
    except Exception as e:
        print(f"\n✗ Error exploring ReMIND: {type(e).__name__}: {e}")


# ============================================================================
# DATASET 3: CAI2R (PET + MRI)
# ============================================================================

def explore_cai2r():
    print("\n" + "="*80)
    print("DATASET 3: CAI2R (PET + MRI Multimodal)")
    print("="*80)
    
    try:
        mat_file = str(PROJECT_ROOT / "Data" / "Raw" / "CAI2R" / "rawdata_mprage_fdg_2013.mat")
        
        with h5py.File(mat_file, 'r') as f:
            print(f"\nOpened HDF5 file: {os.path.basename(mat_file)}")
            print(f"Root keys: {list(f.keys())}")
            
            # Extract MRI data
            print(f"\n1. MRI Data (#refs#/b/data):")
            mri_path = "#refs#/b/data"
            if mri_path in f:
                mri_dataset = f[mri_path]
                mri_data = mri_dataset[()]
                
                print(f"   Raw shape from HDF5: {mri_data.shape}")
                print(f"   Raw dtype: {mri_data.dtype}")
                
                # Convert complex data (real/imag pairs) to complex array
                if mri_data.dtype.names:  # Structured array
                    mri_complex = mri_data['real'] + 1j * mri_data['imag']
                else:
                    mri_complex = mri_data
                
                mri_magnitude = np.abs(mri_complex)
                print(f"   Magnitude shape: {mri_magnitude.shape}")
                print(f"   Magnitude min: {mri_magnitude.min():.4f}")
                print(f"   Magnitude max: {mri_magnitude.max():.4f}")
                print(f"   Magnitude mean: {mri_magnitude.mean():.4f}")
            
            # Extract PET data
            print(f"\n2. PET Data (#refs#/c/data):")
            pet_path = "#refs#/c/data"
            if pet_path in f:
                pet_dataset = f[pet_path]
                pet_data = pet_dataset[()]
                
                print(f"   Shape: {pet_data.shape}")
                print(f"   Dtype: {pet_data.dtype}")
                print(f"   Min: {pet_data.min():.6f}")
                print(f"   Max: {pet_data.max():.6f}")
                print(f"   Mean: {pet_data.mean():.6f}")
                print(f"   Non-zero elements: {np.count_nonzero(pet_data)}")
            
            # Extract readme
            print(f"\n3. Readme Field:")
            readme_path = "#refs#/c/readme"
            if readme_path in f:
                readme_data = f[readme_path][()]
                # Decode uint16 array to string
                try:
                    if isinstance(readme_data, np.ndarray):
                        readme_str = ''.join(chr(int(c)) for c in readme_data.flatten()[:200])
                        print(f"   [First 200 chars] {readme_str}...")
                except Exception as e:
                    print(f"   Could not decode readme: {e}")
            
            # Print scan info summary
            print(f"\n4. PET Scan Info:")
            scan_info_path = "#refs#/c/scan_info"
            if scan_info_path in f:
                scan_info = f[scan_info_path]
                print(f"   Available keys: {list(scan_info.keys())}")
                
                for key in ['frame_duration', 'scatter_fraction']:
                    if key in scan_info:
                        val = scan_info[key][()]
                        print(f"   {key}: {val}")
        
        print(f"\n✓ CAI2R Dataset Summary:")
        print(f"  Contains: MRI (complex) + PET + associated metadata")
        
    except Exception as e:
        print(f"\n✗ Error exploring CAI2R: {type(e).__name__}: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MULTIMODAL BRAIN ABNORMALITY CLASSIFICATION")
    print("Dataset Exploration")
    print("="*80)
    
    explore_brats2020()
    explore_remind()
    explore_cai2r()
    
    print("\n" + "="*80)
    print("Exploration Complete")
    print("="*80 + "\n")
