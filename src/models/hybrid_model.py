"""Advanced Hybrid Skin Lesion Classification Architecture.

Implements a modular multi-stage deep architecture combining:
  1. Backbone Convolutional Feature Extractor (EfficientNetB0, ResNet50, MobileNetV2).
  2. Attention Recalibration (Channel Attention, Spatial Attention, CBAM, Squeeze-and-Excitation).
  3. Residual Attention Shortcut Connection (F_refined = F + Attn(F)).
  4. Clinical Metadata Fusion Branch (Gated Multimodal Unit or Concatenation) when reliable metadata exists.
  5. Multi-layer regularized classification head (Dense projection, BatchNorm, Dropout, Softmax).

Mathematical Formulation:
-------------------------
1. Visual Feature Extraction:
   F = Backbone(X_img) ∈ R^{H × W × C}

2. Adaptive Attention Recalibration:
   - Channel Attention (CAM):
     M_c(F) = σ( W_1(δ(W_0(GAP(F)))) + W_1(δ(W_0(GMP(F)))) )
     F' = M_c(F) ⊗ F
   - Spatial Attention (SAM):
     M_s(F') = σ( f^{7 × 7}([GAP_c(F'); GMP_c(F')]) )
     F'' = M_s(F') ⊗ F'
   - Residual Refinement:
     F_refined = F + F''

3. Global Visual Embedding:
   v_img = GAP(F_refined) ∈ R^C
   h_img = BatchNorm(ReLU(W_img · v_img + b_img)) ∈ R^D

4. Clinical Metadata Embedding (when metadata exists):
   h_meta = BatchNorm(ReLU(W_meta · X_meta + b_meta)) ∈ R^D

5. Multimodal Fusion:
   - Concatenation:
     z = [h_img ; h_meta] ∈ R^{2D}
   - Gated Multimodal Unit (GMU):
     g = σ(W_g · [h_img ; h_meta] + b_g) ∈ R^D
     z = g ⊙ h_img + (1 - g) ⊙ h_meta ∈ R^D

6. Classification Head:
   y_hat = Softmax(W_cls · Dropout(z) + b_cls) ∈ R^{num_classes}
"""

from typing import List, Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras import layers, models

from src.models.attention import CBAM, ChannelAttention, SpatialAttention, SqueezeAndExcitation


def _build_backbone(
    backbone_name: str,
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    freeze_backbone: bool = True,
) -> tf.keras.Model:
    """Build and return pre-trained backbone model without classification head."""
    name = backbone_name.lower().replace("-", "_")

    if "efficientnet" in name:
        from tensorflow.keras.applications import EfficientNetB0
        try:
            base_model = EfficientNetB0(
                include_top=False,
                weights=weights,
                input_shape=input_shape,
            )
        except Exception as exc:
            print(f"[HybridModel] Offline fallback for EfficientNetB0 ({exc}). Initializing with weights=None.")
            base_model = EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)

    elif "resnet" in name:
        from tensorflow.keras.applications import ResNet50
        try:
            base_model = ResNet50(
                include_top=False,
                weights=weights,
                input_shape=input_shape,
            )
        except Exception as exc:
            print(f"[HybridModel] Offline fallback for ResNet50 ({exc}). Initializing with weights=None.")
            base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)

    elif "mobilenet" in name:
        from tensorflow.keras.applications import MobileNetV2
        try:
            base_model = MobileNetV2(
                include_top=False,
                weights=weights,
                input_shape=input_shape,
                alpha=1.0,
            )
        except Exception as exc:
            print(f"[HybridModel] Offline fallback for MobileNetV2 ({exc}). Initializing with weights=None.")
            base_model = MobileNetV2(include_top=False, weights=None, input_shape=input_shape, alpha=1.0)
    else:
        raise ValueError(
            f"Unsupported backbone: '{backbone_name}'. "
            "Supported: 'efficientnet_b0', 'resnet50', 'mobilenet_v2'."
        )

    base_model.trainable = not freeze_backbone
    return base_model


def _apply_attention(
    feature_map: tf.Tensor,
    attention_type: str,
    reduction_ratio: int = 16,
    spatial_kernel_size: int = 7,
    use_residual: bool = True,
) -> tf.Tensor:
    """Apply configured attention mechanism to the spatial feature map."""
    att_key = attention_type.lower().strip()

    if att_key in ("cbam", "dual"):
        attended = CBAM(
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size,
            use_residual=False,  # We manage residual explicitly below
            name="hybrid_cbam_attention",
        )(feature_map)
    elif att_key in ("channel", "cam"):
        attended = ChannelAttention(
            reduction_ratio=reduction_ratio,
            name="hybrid_channel_attention",
        )(feature_map)
    elif att_key in ("spatial", "sam"):
        attended = SpatialAttention(
            kernel_size=spatial_kernel_size,
            name="hybrid_spatial_attention",
        )(feature_map)
    elif att_key in ("se", "squeeze_excitation"):
        attended = SqueezeAndExcitation(
            ratio=reduction_ratio,
            name="hybrid_se_attention",
        )(feature_map)
    elif att_key in ("none", "none_ablation", ""):
        return feature_map
    else:
        raise ValueError(
            f"Unknown attention type '{attention_type}'. "
            "Choose from: 'cbam', 'channel', 'spatial', 'se', 'none'."
        )

    if use_residual:
        return layers.Add(name="residual_attention_add")([feature_map, attended])
    return attended


def build_hybrid_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    backbone: str = "efficientnet_b0",
    weights: Optional[str] = "imagenet",
    freeze_backbone: bool = True,
    attention_type: str = "cbam",
    reduction_ratio: int = 16,
    spatial_kernel_size: int = 7,
    use_residual_attention: bool = True,
    use_metadata: bool = False,
    metadata_dim: int = 10,
    fusion_type: str = "gated",
    dense_units: int = 256,
    dropout_rate: float = 0.35,
    name: str = "DermaAI_Hybrid_Architecture",
) -> tf.keras.Model:
    """Build a modular, research-grade hybrid skin lesion classification model.

    Args:
        input_shape: Shape of dermoscopic input images (H, W, C).
        num_classes: Number of lesion diagnostic categories (default: 7).
        backbone: Feature extractor backbone ('efficientnet_b0', 'resnet50', 'mobilenet_v2').
        weights: Pretrained weights ('imagenet' or None).
        freeze_backbone: Whether to freeze backbone weights during stage 1.
        attention_type: Type of attention ('cbam', 'channel', 'spatial', 'se', 'none').
        reduction_ratio: Channel bottleneck reduction ratio for attention.
        spatial_kernel_size: Spatial convolution kernel size for spatial attention.
        use_residual_attention: Whether to add a residual shortcut F + Attn(F).
        use_metadata: Whether to enable clinical tabular metadata fusion.
        metadata_dim: Feature dimension of the preprocessed clinical metadata vector.
        fusion_type: Fusion mechanism ('gated' for GMU, 'concat' for concatenation).
        dense_units: Dimensionality of the projected visual and metadata embeddings.
        dropout_rate: Regularization dropout probability.
        name: Name identifier for the Keras Model.

    Returns:
        tf.keras.Model: Ready-to-compile modular hybrid classification model.
    """
    # ── 1. Image Input & Vision Backbone ──────────────────────────────────────
    image_input = layers.Input(shape=input_shape, name="image_input")

    # Built-in backbone preprocessing
    if "resnet" in backbone.lower():
        x_prep = tf.keras.applications.resnet50.preprocess_input(image_input)
    elif "mobilenet" in backbone.lower():
        x_prep = tf.keras.applications.mobilenet_v2.preprocess_input(image_input)
    else:
        x_prep = image_input

    base_model = _build_backbone(
        backbone_name=backbone,
        input_shape=input_shape,
        weights=weights,
        freeze_backbone=freeze_backbone,
    )

    feature_map = base_model(x_prep, training=not freeze_backbone)

    # ── 2. Attention Recalibration ─────────────────────────────────────────────
    attended_features = _apply_attention(
        feature_map=feature_map,
        attention_type=attention_type,
        reduction_ratio=reduction_ratio,
        spatial_kernel_size=spatial_kernel_size,
        use_residual=use_residual_attention,
    )

    # ── 3. Visual Feature Projection ──────────────────────────────────────────
    x_img = layers.GlobalAveragePooling2D(name="global_avg_pool")(attended_features)
    x_img = layers.BatchNormalization(name="bn_post_pool")(x_img)
    x_img = layers.Dense(dense_units, kernel_initializer="he_normal", name="img_dense_proj")(x_img)
    x_img = layers.ReLU(name="img_proj_relu")(x_img)
    x_img = layers.Dropout(dropout_rate, name="img_proj_dropout")(x_img)

    # ── 4. Clinical Metadata Fusion Branch (Optional) ─────────────────────────
    if use_metadata:
        metadata_input = layers.Input(shape=(metadata_dim,), name="metadata_input")

        # Multi-layer Perceptron (MLP) for demographic features
        x_meta = layers.Dense(64, kernel_initializer="he_normal", name="meta_dense_1")(metadata_input)
        x_meta = layers.BatchNormalization(name="meta_bn_1")(x_meta)
        x_meta = layers.ReLU(name="meta_relu_1")(x_meta)
        x_meta = layers.Dropout(0.2, name="meta_dropout_1")(x_meta)

        x_meta = layers.Dense(dense_units, kernel_initializer="he_normal", name="meta_dense_proj")(x_meta)
        x_meta = layers.BatchNormalization(name="meta_bn_proj")(x_meta)
        x_meta = layers.ReLU(name="meta_relu_proj")(x_meta)

        # Multimodal Fusion Strategy
        if fusion_type.lower() == "gated":
            # Gated Multimodal Unit (GMU): Dynamic data-dependent routing
            combined = layers.Concatenate(name="fusion_gate_concat")([x_img, x_meta])
            gate = layers.Dense(dense_units, activation="sigmoid", name="gmu_gate_weights")(combined)

            # Complementary gating: g * h_img + (1 - g) * h_meta
            img_gated = layers.Multiply(name="gated_img")([gate, x_img])
            gate_comp = layers.Lambda(lambda g: 1.0 - g, name="gate_complement")(gate)
            meta_gated = layers.Multiply(name="gated_meta")([gate_comp, x_meta])

            fused = layers.Add(name="gmu_fused_features")([img_gated, meta_gated])
        else:
            # Direct feature concatenation
            fused = layers.Concatenate(name="concat_fused_features")([x_img, x_meta])

        head_input = fused
        model_inputs = [image_input, metadata_input]
    else:
        head_input = x_img
        model_inputs = image_input

    # ── 5. Classification Head ────────────────────────────────────────────────
    x = layers.Dense(dense_units // 2, kernel_initializer="he_normal", name="head_dense_intermediate")(head_input)
    x = layers.BatchNormalization(name="head_bn_intermediate")(x)
    x = layers.ReLU(name="head_relu_intermediate")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_final")(x)

    classification_output = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification_output",
    )(x)

    model = models.Model(inputs=model_inputs, outputs=classification_output, name=name)
    return model


# Backward-compatible factory alias
def build_hybrid_attention_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    backbone_weights: Optional[str] = "imagenet",
    cbam_reduction_ratio: int = 16,
    dense_units: int = 256,
    dropout_rate: float = 0.35,
    name: str = "DermaAI_Hybrid_CBAM",
) -> tf.keras.Model:
    """Backward-compatible alias for the CBAM hybrid attention model."""
    return build_hybrid_model(
        input_shape=input_shape,
        num_classes=num_classes,
        backbone="efficientnet_b0",
        weights=backbone_weights,
        attention_type="cbam",
        reduction_ratio=cbam_reduction_ratio,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        use_metadata=False,
        name=name,
    )
