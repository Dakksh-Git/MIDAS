"""
Kaggle Hub Setup and Download Example

Before running this script, set up your Kaggle API credentials:

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" to download kaggle.json
3. Place kaggle.json in: C:\Users\dakks_37ll4de\.kaggle\

Alternatively, set environment variables:
   Set-Env KAGGLE_USERNAME "your-username"
   Set-Env KAGGLE_KEY "your-api-key"
"""

import kagglehub
import os

def setup_credentials():
    """Verify Kaggle credentials are set up"""
    username = os.getenv('KAGGLE_USERNAME')
    api_key = os.getenv('KAGGLE_KEY')
    
    if username and api_key:
        print("✓ Environment variables configured")
        return True
    
    kaggle_file = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_file):
        print("✓ kaggle.json found")
        return True
    
    print("✗ Credentials not found!")
    print("  1. Download API token from: https://www.kaggle.com/settings/account")
    print("  2. Save kaggle.json to: ~/.kaggle/")
    return False

def download_dataset(dataset_path: str, download_dir: str = "datasets"):
    """
    Download a dataset from Kaggle
    
    Args:
        dataset_path: In format "owner/dataset-name" (e.g., "uciml/iris")
        download_dir: Directory to save the dataset
    """
    if not setup_credentials():
        return
    
    print(f"\nDownloading {dataset_path}...")
    path = kagglehub.dataset_download(dataset_path, path=download_dir)
    print(f"✓ Downloaded to: {path}")
    return path

if __name__ == "__main__":
    # Example: Download the Iris dataset
    download_dataset("uciml/iris")
