"""Tests for XAI utilities in DermaAI.

Covers:
 * GradCAM heatmap shape and range
 * Heatmap overlay dimensions
 * DermaExplainer high‑level workflow (prediction & disclaimer)
 * Visualization panel creation
"""

import os
import json
import numpy as np
import tensorflow as tf

from src.explainability.gradcam import GradCAMExplainer
from src.explainability.visualizer import HeatmapVisualizer
from src.explainability.explainer import DermaExplainer, ExplanationResult
from src.explainability.visualization import plot_prediction_panel

# Simple tiny model for deterministic test – single Conv2D + GlobalAvg + Dense
def _build_test_model(num_classes: int = 3, input_shape=(32, 32, 3)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", name="conv_test")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs)


def test_gradcam_heatmap_shape_and_range(tmp_path):
    model = _build_test_model()
    expl = GradCAMExplainer(model)
    # Random image tensor
    img = np.random.rand(1, 32, 32, 3).astype(np.float32)
    heatmap = expl.generate_heatmap(img)
    assert heatmap.shape == (30, 30)  # conv layer reduces size by 2 (valid padding default)
    assert np.all(heatmap >= 0) and np.all(heatmap <= 1)


def test_heatmap_overlay_dimensions(tmp_path):
    # Dummy image and heatmap
    img = np.random.rand(32, 32, 3)
    heat = np.random.rand(30, 30)
    overlay = HeatmapVisualizer.overlay_heatmap(img, heat)
    assert overlay.shape == img.shape
    assert overlay.dtype == np.uint8


def test_derma_explainer_full_flow(tmp_path):
    model = _build_test_model()
    checkpoint_path = tmp_path / "model.keras"
    model.save(str(checkpoint_path), save_format="keras")
    explainer = DermaExplainer(str(checkpoint_path), class_names=["A", "B", "C"])
    # Create a small dummy image file for the explainer to load
    img_arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    img_path = tmp_path / "img.png"
    tf.keras.utils.save_img(str(img_path), img_arr)
    result: ExplanationResult = explainer.explain(str(img_path))
    # Basic sanity checks
    assert isinstance(result, ExplanationResult)
    assert result.heatmap.shape[0] > 0
    assert result.overlay.shape == img_arr.shape
    assert result.disclaimer.startswith("⚠")
    # Save explanation and verify artefacts
    out_dir = tmp_path / "out"
    explainer.save_explanation(result, str(out_dir))
    base = os.path.splitext(os.path.basename(str(img_path)))[0]
    assert (out_dir / base / "overlay.png").exists()
    manifest = json.loads((out_dir / base / "manifest.json").read_text())
    assert manifest["predicted_class_name"] in ["A", "B", "C"]


def test_plot_prediction_panel(tmp_path):
    img = np.random.rand(32, 32, 3)
    heat = np.random.rand(30, 30)
    overlay = HeatmapVisualizer.overlay_heatmap(img, heat)
    probs = np.array([0.1, 0.7, 0.2])
    fig = plot_prediction_panel(
        image=img,
        heatmap=heat,
        overlay=overlay,
        class_probs=probs,
        class_names=["A", "B", "C"],
        predicted_class=1,
        save_path=str(tmp_path / "panel.png"),
    )
    assert fig is not None
    assert (tmp_path / "panel.png").exists()
