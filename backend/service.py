"""Model Inference Service Orchestrator for DermaAI Backend."""

import base64
import cv2
import io
import time
import os

# Existing imports ... (the top part will be inserted after the import statements)
# Insert after line 7 (which currently is "from typing import Dict, List, Optional, Tuple")


import numpy as np
import tensorflow as tf
from PIL import Image

from src.models.transfer_models import build_transfer_model
from src.preprocessing.image_cleaner import DullRazorCleaner, GrayWorldColorConstancy
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.visualizer import HeatmapVisualizer
from src.utils import setup_logger
import io
import time
import os

# Existing imports ... (the top part will be inserted after the import statements)
# Insert after line 7 (which currently is "from typing import Dict, List, Optional, Tuple")

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from src.models.transfer_models import build_transfer_model
from src.preprocessing.image_cleaner import DullRazorCleaner, GrayWorldColorConstancy
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.visualizer import HeatmapVisualizer
from src.utils import setup_logger

logger = setup_logger("DermaAI.InferenceService")

CLASS_METADATA = [
    {"code": "akiec", "name": "Actinic Keratoses and Intraepithelial Carcinoma", "risk_level": "High"},
    {"code": "bcc", "name": "Basal Cell Carcinoma", "risk_level": "High"},
    {"code": "bkl", "name": "Benign Keratosis-like Lesions", "risk_level": "Low"},
    {"code": "df", "name": "Dermatofibroma", "risk_level": "Low"},
    {"code": "mel", "name": "Melanoma", "risk_level": "Critical"},
    {"code": "nv", "name": "Melanocytic Nevi", "risk_level": "Low"},
    {"code": "vasc", "name": "Vascular Lesions", "risk_level": "Low"},
]


class ModelInferenceService:
    """Manages model lifecycle, image preprocessing, forward pass execution, and Grad-CAM generation."""

    _instance = None

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.hair_cleaner = DullRazorCleaner()
        self.color_normalizer = GrayWorldColorConstancy()
        self.model: Optional[tf.keras.Model] = None
        self.explainer: Optional[GradCAMExplainer] = None
        self.backbone_name = "efficientnet_b0"
        self._load_or_initialize_model()

    @classmethod
    def get_instance(cls) -> "ModelInferenceService":
        """Singleton accessor for inference service."""
        if cls._instance is None:
            cls._instance = ModelInferenceService()
        return cls._instance

    def _load_or_initialize_model(self) -> None:
        """Initialize transfer learning model architecture."""
        try:
            logger.info("Initializing EfficientNetB0 feature extraction model for inference...")
            self.model = build_transfer_model(
                backbone_name=self.backbone_name,
                input_shape=(self.target_size[0], self.target_size[1], 3),
                num_classes=len(CLASS_METADATA),
                weights="imagenet",
                freeze_base=True,
            )
            # Create warm-up inference pass
            dummy_input = tf.zeros((1, self.target_size[0], self.target_size[1], 3))
            _ = self.model(dummy_input)
            self.explainer = GradCAMExplainer(self.model)
            logger.info("Inference model and Grad-CAM explainer ready.")
        except Exception as e:
            logger.error(f"Error initializing inference model: {e}")
            raise

    def preprocess_image(self, image_bytes: bytes, apply_dull_razor: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Convert raw uploaded image bytes into preprocessed RGB arrays."""
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)

        # Hair removal if requested
        if apply_dull_razor:
            img_np = self.hair_cleaner.remove_hair(img_np)

        # Color constancy illumination correction
        img_np = self.color_normalizer.apply(img_np)

        # Resize to network resolution
        resized = cv2.resize(img_np, self.target_size, interpolation=cv2.INTER_AREA)
        tensor_input = (resized.astype(np.float32) / 255.0)[np.newaxis, ...]
        return img_np, tensor_input

    def predict(
        self,
        image_bytes: bytes,
        include_gradcam: bool = True,
        apply_dull_razor: bool = True,
    ) -> Dict:
        """Execute full inference pipeline and return top-k diagnoses and Grad-CAM visualization."""
        start_time = time.time()
        orig_img, tensor_input = self.preprocess_image(image_bytes, apply_dull_razor=apply_dull_razor)

        # Model forward pass
        probs = self.model(tensor_input).numpy()[0]
        top_idx = int(np.argmax(probs))

        predictions = []
        for idx, prob in enumerate(probs):
            item = CLASS_METADATA[idx].copy()
            item["probability"] = float(prob)
            predictions.append(item)

        # Sort descending by probability
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        top_prediction = predictions[0]

        # Clinical high risk flag
        high_risk = top_prediction["code"] in ["mel", "bcc", "akiec"]

        # Explainability: Grad-CAM
        gradcam_base64 = None
        if include_gradcam and self.explainer:
            try:
                heatmap = self.explainer.generate_heatmap(tensor_input, class_idx=top_idx)
                blended = HeatmapVisualizer.overlay_heatmap(cv2.resize(orig_img, self.target_size), heatmap)
                # Encode to Base64
                is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                if is_success:
                    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                logger.warning(f"Grad-CAM generation encountered error: {e}")

        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "top_prediction": top_prediction,
            "predictions": predictions,
            "high_risk_alert": high_risk,
            "gradcam_base64": gradcam_base64,
            "inference_time_ms": round(elapsed_ms, 2),
        }
