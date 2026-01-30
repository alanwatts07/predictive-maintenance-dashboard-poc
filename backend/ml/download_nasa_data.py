"""
Download and extract NASA IMS Bearing Dataset.

Dataset: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
Mirror: https://data.nasa.gov/download/brfb-gzcv/application%2Fzip

The dataset contains run-to-failure vibration data from bearing tests:
- Test 1: 2003.10.22 - 2003.11.25 (34 days, bearing 3 inner race failure)
- Test 2: 2003.12.08 - 2004.02.19 (73 days, bearing 1 outer race failure)
- Test 3: 2004.03.04 - 2004.04.18 (45 days, bearing 3 outer race failure)
"""

import os
import zipfile
import urllib.request
import shutil
from pathlib import Path

# NASA IMS dataset mirrors
# Primary: Kaggle dataset (most reliable)
# Requires kaggle CLI or manual download from: https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
BACKUP_URL = "https://ti.arc.nasa.gov/m/project/prognostic-repository/IMS.zip"

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def download_dataset(url: str = DATASET_URL) -> Path:
    """Download the NASA bearing dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR / "nasa_bearing.zip"

    if zip_path.exists():
        print(f"Dataset already downloaded: {zip_path}")
        return zip_path

    print(f"Downloading NASA IMS Bearing Dataset...")
    print(f"URL: {url}")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to: {zip_path}")
    except Exception as e:
        print(f"Primary URL failed: {e}")
        print(f"Trying backup URL...")
        urllib.request.urlretrieve(BACKUP_URL, zip_path)

    return zip_path


def extract_dataset(zip_path: Path) -> Path:
    """Extract the dataset."""
    if (RAW_DIR / "1st_test").exists() or (RAW_DIR / "2nd_test").exists():
        print(f"Dataset already extracted to: {RAW_DIR}")
        return RAW_DIR

    print(f"Extracting dataset...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(RAW_DIR)

    # Handle nested directory structure from GitHub
    nested = RAW_DIR / "datasets-nasa-bearing-main"
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(RAW_DIR / item.name))
        nested.rmdir()

    print(f"Extracted to: {RAW_DIR}")
    return RAW_DIR


def setup_data():
    """Download and extract the NASA bearing dataset."""
    zip_path = download_dataset()
    extract_dataset(zip_path)
    print("\nDataset ready!")
    print(f"Raw data: {RAW_DIR}")
    return RAW_DIR


if __name__ == "__main__":
    setup_data()
