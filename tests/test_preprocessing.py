"""Unit Tests for Image Preprocessing and Clinical Tabular Feature Transformers."""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing.image_cleaner import DullRazorCleaner, GrayWorldColorConstancy
from src.preprocessing.tabular_preprocessor import ClinicalTabularPreprocessor


def test_dull_razor_cleaner_shape_and_range():
    """Verify DullRazor output preserves image shape, uint8 type, and channel order."""
    cleaner = DullRazorCleaner(filter_kernel_size=9, threshold=10, inpaint_radius=5)
    # Generate dummy skin image with simulated dark hair lines
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 200
    # Add synthetic hair line
    dummy_image[45:55, :, :] = 20

    cleaned = cleaner.remove_hair(dummy_image)
    assert cleaned.shape == dummy_image.shape
    assert cleaned.dtype == np.uint8
    # Assert hair region was inpainted to brighter tone
    assert np.mean(cleaned[45:55, :, :]) > np.mean(dummy_image[45:55, :, :])


def test_gray_world_color_constancy():
    """Verify Gray World algorithm normalizes color channels."""
    # Create image with artificial red color cast
    biased_image = np.zeros((50, 50, 3), dtype=np.uint8)
    biased_image[:, :, 0] = 220 # High Red
    biased_image[:, :, 1] = 60  # Low Green
    biased_image[:, :, 2] = 60  # Low Blue

    balanced = GrayWorldColorConstancy.apply(biased_image)
    assert balanced.shape == biased_image.shape
    # Channels should become much more balanced towards equal values
    r_mean = np.mean(balanced[:, :, 0])
    g_mean = np.mean(balanced[:, :, 1])
    assert abs(r_mean - g_mean) < abs(220 - 60)


def test_clinical_tabular_preprocessor():
    """Verify demographic feature transformer handles missing values, scaling, and one-hot encoding."""
    raw_data = pd.DataFrame({
        "age": [55.0, np.nan, 30.0, 72.0],
        "sex": ["male", "female", "unknown", "male"],
        "localization": ["back", "face", "trunk", "lower extremity"],
    })

    preprocessor = ClinicalTabularPreprocessor(
        numerical_cols=["age"],
        categorical_cols=["sex", "localization"],
    )

    features = preprocessor.fit_transform(raw_data)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 4
    assert features.shape[1] == preprocessor.output_dim
    assert not np.isnan(features).any()
