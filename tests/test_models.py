"""Unit Tests for Deep Learning Architectures and Attention Layers."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.models.baseline_cnn import build_baseline_cnn
from src.models.resnet_model import build_resnet50
from src.models.efficientnet_model import build_efficientnet_b0
from src.models.mobilenet_model import build_mobilenet_v2
from src.models.transfer_models import build_transfer_model, get_available_backbones
from src.models.attention import CBAM, ChannelAttention, SpatialAttention, SqueezeAndExcitation
from src.models.hybrid_model import build_hybrid_attention_model


def test_baseline_cnn_forward_pass():
    """Verify custom baseline CNN forward pass and output probability distribution."""
    model = build_baseline_cnn(
        input_shape=(64, 64, 3),
        num_classes=7,
        conv_filters=[16, 32],
        dense_units=[64],
        l2_reg=1e-4,
    )
    dummy_input = tf.random.uniform((2, 64, 64, 3))
    output = model(dummy_input)

    assert output.shape == (2, 7)
    np.testing.assert_allclose(tf.reduce_sum(output, axis=-1).numpy(), [1.0, 1.0], rtol=1e-5)
    print("PASS: test_baseline_cnn_forward_pass")


def test_resnet50_forward_pass():
    """Verify ResNet50 transfer learning model instantiates and infers."""
    model = build_resnet50(
        input_shape=(64, 64, 3),
        num_classes=7,
        weights=None,
        freeze_base=True,
    )
    dummy_input = tf.random.uniform((2, 64, 64, 3))
    output = model(dummy_input)
    assert output.shape == (2, 7)
    print("PASS: test_resnet50_forward_pass")


def test_efficientnet_forward_pass():
    """Verify EfficientNetB0 transfer learning model instantiates and infers."""
    model = build_efficientnet_b0(
        input_shape=(64, 64, 3),
        num_classes=7,
        weights=None,
        freeze_base=True,
    )
    dummy_input = tf.random.uniform((2, 64, 64, 3))
    output = model(dummy_input)
    assert output.shape == (2, 7)
    print("PASS: test_efficientnet_forward_pass")


def test_mobilenet_forward_pass():
    """Verify MobileNetV2 transfer learning model instantiates and infers."""
    model = build_mobilenet_v2(
        input_shape=(64, 64, 3),
        num_classes=7,
        weights=None,
        freeze_base=True,
    )
    dummy_input = tf.random.uniform((2, 64, 64, 3))
    output = model(dummy_input)
    assert output.shape == (2, 7)
    print("PASS: test_mobilenet_forward_pass")


def test_channel_attention_layer():
    """Verify ChannelAttention preserves spatial dimensions and learns weights."""
    ca = ChannelAttention(reduction_ratio=4)
    dummy_feature_map = tf.random.uniform((2, 8, 8, 16))
    attended = ca(dummy_feature_map)
    assert attended.shape == dummy_feature_map.shape
    print("PASS: test_channel_attention_layer")


def test_spatial_attention_layer():
    """Verify SpatialAttention preserves spatial dimensions."""
    sa = SpatialAttention(kernel_size=3)
    dummy_feature_map = tf.random.uniform((2, 8, 8, 16))
    attended = sa(dummy_feature_map)
    assert attended.shape == dummy_feature_map.shape
    print("PASS: test_spatial_attention_layer")


def test_cbam_attention_layer():
    """Verify CBAM layer preserves spatial tensor dimensions."""
    cbam = CBAM(reduction_ratio=4, spatial_kernel_size=3)
    dummy_feature_map = tf.random.uniform((2, 8, 8, 16))
    attended = cbam(dummy_feature_map)
    assert attended.shape == dummy_feature_map.shape
    print("PASS: test_cbam_attention_layer")


def test_se_attention_layer():
    """Verify SqueezeAndExcitation layer preserves tensor dimensions."""
    se = SqueezeAndExcitation(ratio=4)
    dummy_feature_map = tf.random.uniform((2, 8, 8, 16))
    attended = se(dummy_feature_map)
    assert attended.shape == dummy_feature_map.shape
    print("PASS: test_se_attention_layer")


def test_transfer_model_builder():
    """Verify transfer learning backbone factory instantiates with correct output shape."""
    model = build_transfer_model(
        backbone_name="efficientnet_b0",
        input_shape=(64, 64, 3),
        num_classes=7,
        weights=None,
    )
    dummy_input = tf.random.uniform((1, 64, 64, 3))
    preds = model(dummy_input)
    assert preds.shape == (1, 7)
    print("PASS: test_transfer_model_builder")


if __name__ == "__main__":
    tests = [
        test_baseline_cnn_forward_pass,
        test_resnet50_forward_pass,
        test_efficientnet_forward_pass,
        test_mobilenet_forward_pass,
        test_channel_attention_layer,
        test_spatial_attention_layer,
        test_cbam_attention_layer,
        test_se_attention_layer,
        test_transfer_model_builder,
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
        import sys
        sys.exit(1)
