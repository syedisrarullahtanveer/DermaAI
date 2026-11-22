"""Unit Tests for Patient-Aware Data Splitter and Leakage Verifier."""

from pathlib import Path
import pandas as pd

from src.data.splitter import PatientAwareDataSplitter


def test_patient_aware_splitter_zero_leakage():
    """Verify that multiple images of the same lesion never cross split boundaries."""
    # Create dataset with multiple images per lesion (e.g. L001 has 3 images, L002 has 2 images)
    data = []
    for i in range(40):
        lesion_id = f"LESION_{i:03d}"
        diagnosis = ["nv", "mel", "bcc", "bkl"][i % 4]
        # Some lesions have 2 or 3 images
        num_imgs = 1 if i % 3 != 0 else 3
        for j in range(num_imgs):
            data.append({
                "image_id": f"IMG_{i:03d}_{j}",
                "lesion_id": lesion_id,
                "dx": diagnosis,
            })

    df = pd.DataFrame(data)
    splitter = PatientAwareDataSplitter(df, seed=42)
    train_df, val_df, test_df = splitter.split(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # 1. Total samples must match
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # 2. Assert zero patient overlap
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    assert len(train_lesions.intersection(val_lesions)) == 0
    assert len(train_lesions.intersection(test_lesions)) == 0
    assert len(val_lesions.intersection(test_lesions)) == 0


def test_splitter_export(tmp_path):
    """Verify split CSV manifests are exported correctly."""
    df = pd.DataFrame({
        "image_id": [f"IMG_{i}" for i in range(20)],
        "lesion_id": [f"LESION_{i}" for i in range(20)],
        "dx": ["nv"] * 10 + ["mel"] * 10,
    })
    splitter = PatientAwareDataSplitter(df, seed=42)
    train_df, val_df, test_df = splitter.split()

    paths = splitter.export_splits(train_df, val_df, test_df, output_dir=tmp_path)
    assert Path(paths["train"]).exists()
    assert Path(paths["val"]).exists()
    assert Path(paths["test"]).exists()
