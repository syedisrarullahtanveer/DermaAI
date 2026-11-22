"""Unit Tests for Dermoscopy Augmentations, CutMix, and MixUp."""

import numpy as np
from src.preprocessing.augmentations import DermaAugmentationPipeline


def test_derma_augmentation_pipeline_numpy():
    """Verify augment_numpy returns valid dimensions and preserves dtypes."""
    pipeline = DermaAugmentationPipeline(seed=42)

    # Test uint8 image
    test_img_uint8 = np.ones((100, 100, 3), dtype=np.uint8) * 150
    aug_uint8 = pipeline.augment_numpy(test_img_uint8)
    assert aug_uint8.shape == (100, 100, 3)
    assert aug_uint8.dtype == np.uint8

    # Test float32 image
    test_img_float = np.ones((80, 80, 3), dtype=np.float32) * 0.6
    aug_float = pipeline.augment_numpy(test_img_float)
    assert aug_float.shape == (80, 80, 3)
    assert aug_float.dtype == np.float32
    assert aug_float.min() >= 0.0 and aug_float.max() <= 1.0


def test_mixup_and_cutmix():
    """Verify MixUp and CutMix blending and label distributions."""
    img1 = np.ones((64, 64, 3), dtype=np.float32) * 0.2
    img2 = np.ones((64, 64, 3), dtype=np.float32) * 0.8
    lbl1 = np.array([1.0, 0.0, 0.0]) # Class 0
    lbl2 = np.array([0.0, 1.0, 0.0]) # Class 1

    # MixUp
    mix_img, mix_lbl = DermaAugmentationPipeline.mixup(img1, img2, lbl1, lbl2, alpha=0.5)
    assert mix_img.shape == (64, 64, 3)
    assert mix_lbl.shape == (3,)
    np.testing.assert_allclose(np.sum(mix_lbl), 1.0, rtol=1e-5)

    # CutMix
    cut_img, cut_lbl = DermaAugmentationPipeline.cutmix(img1, img2, lbl1, lbl2, alpha=1.0)
    assert cut_img.shape == (64, 64, 3)
    assert cut_lbl.shape == (3,)
    np.testing.assert_allclose(np.sum(cut_lbl), 1.0, rtol=1e-5)
