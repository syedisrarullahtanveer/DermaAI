"""Unit Tests for Dataset Distribution and Demographic Analyzer."""

import pandas as pd
from src.data.dataset_analyzer import DatasetDistributionAnalyzer


def test_dataset_distribution_analyzer():
    """Verify class distribution, imbalance ratio, and demographic calculations."""
    sample_df = pd.DataFrame({
        "image_id": [f"IMG_{i:03d}" for i in range(10)],
        "lesion_id": ["L1", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"],
        "dx": ["nv", "nv", "nv", "nv", "nv", "nv", "mel", "mel", "bcc", "akiec"],
        "age": [45.0, 45.0, 60.0, 30.0, None, 70.0, 52.0, 65.0, 58.0, 40.0],
        "sex": ["male", "male", "female", "male", "female", "unknown", "female", "male", "male", "female"],
        "localization": ["back", "back", "face", "trunk", "lower extremity", "face", "back", "trunk", "face", "neck"],
    })

    analyzer = DatasetDistributionAnalyzer(sample_df)
    analysis = analyzer.generate_full_analysis_dict()

    # Verify class distribution
    class_dist = analysis["class_distribution"]
    assert class_dist["total_samples"] == 10
    assert class_dist["counts"]["nv"] == 6
    assert class_dist["counts"]["mel"] == 2
    assert class_dist["counts"]["bcc"] == 1
    assert class_dist["majority_class"] == "nv"
    assert class_dist["minority_class"] in ["bcc", "akiec"]
    # Imbalance ratio: 6 / 1 = 6.0
    assert class_dist["imbalance_ratio"] == 6.0

    # Verify patient multiplicity
    mult = analysis["patient_multiplicity"]
    assert mult["total_records"] == 10
    assert mult["unique_lesions"] == 9
    assert mult["multi_image_lesions"] == 1 # L1 has 2 images

    # Verify demographics
    demo = analysis["demographics"]
    assert demo["age"]["missing_samples"] == 1
    assert demo["sex"]["male"] == 5
    assert demo["sex"]["female"] == 4
