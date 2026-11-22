"""Unit Tests for Dataset Integrity and Corruption Checker."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessing.integrity_checker import DatasetIntegrityChecker


def test_integrity_checker_missing_and_orphan(tmp_path):
    """Verify integrity checker accurately identifies missing and orphan images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create 2 valid images on disk
    img1_path = img_dir / "ISIC_0001.jpg"
    img2_path = img_dir / "ISIC_0002.jpg"
    orphan_path = img_dir / "ISIC_9999.jpg"

    dummy_rgb = Image.fromarray((np.ones((30, 30, 3)) * 128).astype(np.uint8))
    dummy_rgb.save(img1_path)
    dummy_rgb.save(img2_path)
    dummy_rgb.save(orphan_path)

    # Metadata lists ISIC_0001, ISIC_0002, and ISIC_0003 (which is missing from disk)
    metadata_df = pd.DataFrame({
        "image_id": ["ISIC_0001", "ISIC_0002", "ISIC_0003"],
        "lesion_id": ["HAM_1", "HAM_2", "HAM_3"],
        "dx": ["mel", "nv", "bcc"],
    })

    checker = DatasetIntegrityChecker(metadata_df=metadata_df, image_dirs=[img_dir])

    # Check missing
    missing = checker.check_missing_images()
    assert missing == ["ISIC_0003"]

    # Check orphan
    orphans = checker.check_orphan_images()
    assert orphans == ["ISIC_9999"]


def test_integrity_checker_detects_corrupted_file(tmp_path):
    """Verify corruption detection identifies zero-byte and truncated image files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create 1 valid image
    valid_path = img_dir / "valid_001.jpg"
    dummy_rgb = Image.fromarray((np.ones((20, 20, 3)) * 150).astype(np.uint8))
    dummy_rgb.save(valid_path)

    # Create 1 zero-byte file
    zero_byte_path = img_dir / "zero_byte.jpg"
    zero_byte_path.touch()

    # Create 1 corrupted header file
    corrupt_path = img_dir / "corrupted.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_STREAM_123456789")

    metadata_df = pd.DataFrame({
        "image_id": ["valid_001", "zero_byte", "corrupted"],
        "lesion_id": ["L1", "L2", "L3"],
        "dx": ["nv", "nv", "mel"],
    })

    checker = DatasetIntegrityChecker(metadata_df=metadata_df, image_dirs=[img_dir])
    audit = checker.run_full_audit()

    assert audit["valid_files_count"] == 1
    assert audit["corrupted_files_count"] == 2
    corrupt_errors = [item["error"] for item in audit["corrupted_files"]]
    assert "ZeroByteFile" in corrupt_errors
