"""Unit Tests for Modular Advanced Hybrid Skin Lesion Classification Architecture."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.models.hybrid_model import build_hybrid_model, build_hybrid_attention_model


def test_hybrid_cbam_vision_only():
    """Verify hybrid model with CBAM attention (vision-only) forward pass."""
    model = build_hybrid_model(
        input_shape=(64, 64, 3),
        num_classes=7,
        backbone="efficientnet_b0",
        weights=None,
        attention_type="cbam",
        use_residual_attention=True,
        use_metadata=False,
    )
    dummy_imgs = tf.random.uniform((2, 64, 64, 3))
    preds = model(dummy_imgs)
    assert preds.shape == (2, 7)
    np.testing.assert_allclose(tf.reduce_sum(preds, axis=-1).numpy(), [1.0, 1.0], rtol=1e-5)
    print("PASS: test_hybrid_cbam_vision_only")


def test_hybrid_channel_attention_ablation():
    """Verify hybrid model with Channel Attention only (Ablation 1)."""
    model = build_hybrid_model(
        input_shape=(64, 64, 3),
        num_classes=7,
        backbone="efficientnet_b0",
        weights=None,
        attention_type="channel",
        use_metadata=False,
    )
    dummy_imgs = tf.random.uniform((2, 64, 64, 3))
    preds = model(dummy_imgs)
    assert preds.shape == (2, 7)
    print("PASS: test_hybrid_channel_attention_ablation")


def test_hybrid_spatial_attention_ablation():
    """Verify hybrid model with Spatial Attention only (Ablation 2)."""
    model = build_hybrid_model(
        input_shape=(64, 64, 3),
        num_classes=7,
        backbone="efficientnet_b0",
        weights=None,
        attention_type="spatial",
        use_metadata=False,
    )
    dummy_imgs = tf.random.uniform((2, 64, 64, 3))
    preds = model(dummy_imgs)
    assert preds.shape == (2, 7)
    print("PASS: test_hybrid_spatial_attention_ablation")


def test_hybrid_no_attention_ablation():
    """Verify hybrid model with No Attention (Ablation 3)."""
    model = build_hybrid_model(
        input_shape=(64, 64, 3),
        num_classes=7,
        backbone="efficientnet_b0",
        weights=None,
        attention_type="none",
        use_metadata=False,
    )
    dummy_imgs = tf.random.uniform((2, 64, 64, 3))
    preds = model(dummy_imgs)
    assert preds.shape == (2, 7)
    print("PASS: test_hybrid_no_attention_ablation")


def test_hybrid_se_attention():
    """Verify hybrid model with Squeeze-and-Excitation attention."""
    model = build_hybrid_model(
        input_shape=(64, 64, 3),
        num_classes=7,
        backbone="efficientnet_b0",
        weights=None,
        attention_type="se",
        use_metadata=False,
    )
    dummy_imgs = tf.random.uniform((2, 64, 64, 3))
    preds = model(dummy_imgs)
    assert preds.shape == (2, 7)
    print("PASS: test_hybrid_se_attention")


def test_hybrid_multimodal_concat():
    """Verify multimodal hybrid model with feature concatenation."""
    meta_dim = 10
    model = build_hybrid_model(
        input_shape=(64, 64, 3),
        num_classes=7,
        backbone="efficientnet_b0",
        weights=None,
        attention_type="cbam",
        use_metadata=True,
        metadata_dim=meta_dim,
        fusion_type="concat",
    )
    dummy_imgs = tf.random.uniform((2, 64, 64, 3))
    dummy_meta = tf.random.normal((2, meta_dim))
    preds = model([dummy_imgs, dummy_meta])
    assert preds.shape == (2, 7)
    np.testing.assert_allclose(tf.reduce_sum(preds, axis=-1).numpy(), [1.0, 1.0], rtol=1e-5)
    print("PASS: test_hybrid_multimodal_concat")


def test_hybrid_multimodal_gated():
    """Verify multimodal hybrid model with Gated Multimodal Unit (GMU)."""
    meta_dim = 10
    model = build_hybrid_model(
        input_shape=(64, 64, 3),
        num_classes=7,
        backbone="efficientnet_b0",
        weights=None,
        attention_type="cbam",
        use_metadata=True,
        metadata_dim=meta_dim,
        fusion_type="gated",
    )
    dummy_imgs = tf.random.uniform((2, 64, 64, 3))
    dummy_meta = tf.random.normal((2, meta_dim))
    preds = model([dummy_imgs, dummy_meta])
    assert preds.shape == (2, 7)
    np.testing.assert_allclose(tf.reduce_sum(preds, axis=-1).numpy(), [1.0, 1.0], rtol=1e-5)
    print("PASS: test_hybrid_multimodal_gated")


def test_hybrid_resnet_backbone():
    """Verify hybrid architecture works with ResNet50 backbone."""
    model = build_hybrid_model(
        input_shape=(64, 64, 3),
        num_classes=7,
        backbone="resnet50",
        weights=None,
        attention_type="cbam",
        use_metadata=False,
    )
    dummy_imgs = tf.random.uniform((2, 64, 64, 3))
    preds = model(dummy_imgs)
    assert preds.shape == (2, 7)
    print("PASS: test_hybrid_resnet_backbone")


def test_backward_compatible_alias():
    """Verify build_hybrid_attention_model alias works."""
    model = build_hybrid_attention_model(
        input_shape=(64, 64, 3),
        num_classes=7,
        backbone_weights=None,
    )
    dummy_imgs = tf.random.uniform((1, 64, 64, 3))
    preds = model(dummy_imgs)
    assert preds.shape == (1, 7)
    print("PASS: test_backward_compatible_alias")


if __name__ == "__main__":
    tests = [
        test_hybrid_cbam_vision_only,
        test_hybrid_channel_attention_ablation,
        test_hybrid_spatial_attention_ablation,
        test_hybrid_no_attention_ablation,
        test_hybrid_se_attention,
        test_hybrid_multimodal_concat,
        test_hybrid_multimodal_gated,
        test_hybrid_resnet_backbone,
        test_backward_compatible_alias,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
