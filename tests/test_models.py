"""Unit Tests for Deep Learning Architectures and Attention Layers."""

import numpy as np
import pytest
import tensorflow as tf

from src.models.baseline_cnn import build_baseline_cnn
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
    )
    dummy_input = tf.random.uniform((2, 64, 64, 3))
    output = model(dummy_input)

    assert output.shape == (2, 7)
    # Probabilities must sum to 1.0 (softmax output)
    np.testing.assert_allclose(tf.reduce_sum(output, axis=-1).numpy(), [1.0, 1.0], rtol=1e-5)


def test_cbam_attention_layer():
    """Verify CBAM layer preserves spatial tensor dimensions."""
    cbam = CBAM(reduction_ratio=8, spatial_kernel_size=7)
    dummy_feature_map = tf.random.uniform((2, 14, 14, 32))
    attended = cbam(dummy_feature_map)

    assert attended.shape == dummy_feature_map.shape


def test_transfer_model_builder():
    """Verify transfer learning backbone factory instantiates with correct output shape."""
    model = build_transfer_model(
        backbone_name="efficientnet_b0",
        input_shape=(64, 64, 3),
        num_classes=7,
        weights=None, # None for fast unit testing
    )
    dummy_input = tf.random.uniform((1, 64, 64, 3))
    preds = model(dummy_input)

    assert preds.shape == (1, 7)
