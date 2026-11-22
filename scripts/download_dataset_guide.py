"""Dataset Acquisition and Licensing Verification Guide.

Provides instructions and verification checks for acquiring the HAM10000 dataset
in strict compliance with Creative Commons CC BY-NC 4.0 terms.
"""

import os
import sys
from pathlib import Path


def print_download_guide():
    print("=" * 70)
    print("  DermaAI: HAM10000 Dataset Acquisition Guide & License Terms")
    print("=" * 70)
    print("""
The HAM10000 ("Human Against Machine with 10,000 training images") dataset
is distributed under the Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) license.

To download the official dataset:

METHOD 1: Official Harvard Dataverse (Direct Download)
------------------------------------------------------
1. Navigate to: https://doi.org/10.7910/DVN/DBW86T
2. Download the following required files:
   - HAM10000_metadata.csv
   - HAM10000_images_part_1.zip
   - HAM10000_images_part_2.zip
3. Extract the contents into the DermaAI project directory:
   - Place 'HAM10000_metadata.csv' -> 'dataset/metadata/HAM10000_metadata.csv'
   - Extract images into:
     Option A (Separate): 'dataset/raw/HAM10000_images_part_1/' and 'dataset/raw/HAM10000_images_part_2/'
     Option B (Merged):   'dataset/raw/HAM10000_images/'

METHOD 2: Kaggle CLI (Automated API Download)
---------------------------------------------
If you have Kaggle CLI configured (~/.kaggle/kaggle.json):
   $ kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p dataset/raw/
   $ unzip dataset/raw/skin-cancer-mnist-ham10000.zip -d dataset/raw/
   $ mv dataset/raw/HAM10000_metadata.csv dataset/metadata/

COMPLIANCE WARNING:
- Raw medical datasets must NEVER be checked into public Git repositories.
- The '.gitignore' in this project is strictly configured to protect against accidental commits.
- For local testing without downloading 3GB of data, run:
   $ python scripts/create_sample_dataset.py
""")
    print("=" * 70)


def verify_local_dataset(project_root: Path = None):
    """Check if HAM10000 is present in the local filesystem."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    metadata_path = project_root / "dataset" / "metadata" / "HAM10000_metadata.csv"
    raw_dir = project_root / "dataset" / "raw"

    print(f"Checking for local dataset at: {project_root}")
    if metadata_path.exists():
        print(f" [x] Metadata found: {metadata_path}")
    else:
        print(f" [ ] Metadata missing: Expected at {metadata_path}")

    image_count = 0
    if raw_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_count += len(list(raw_dir.rglob(ext)))

    print(f" [x] Found {image_count} raw images under {raw_dir}")
    if image_count >= 10000:
        print(" [x] Full HAM10000 dataset is present and verified.")
    elif image_count > 0:
        print(f" [i] Partial dataset detected ({image_count} images).")
    else:
        print(" [i] No raw images found. You can generate sample data using: python scripts/create_sample_dataset.py")


if __name__ == "__main__":
    print_download_guide()
    verify_local_dataset()
