"""Pydantic Request and Response Schemas for DermaAI REST API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClinicalMetadataInput(BaseModel):
    """Patient clinical demographic inputs accompanying dermoscopy scan."""
    age: Optional[float] = Field(default=None, description="Patient age in years", ge=0, le=120)
    sex: Optional[str] = Field(default="unknown", description="Patient biological sex ('male', 'female', 'unknown')")
    localization: Optional[str] = Field(default="unknown", description="Anatomical site where lesion appeared")


class ClassPrediction(BaseModel):
    """Individual diagnosis prediction score."""
    code: str = Field(description="Diagnosis short code (e.g. 'mel', 'bcc')")
    name: str = Field(description="Full clinical name of the lesion")
    probability: float = Field(description="Confidence probability in range [0, 1]")
    risk_level: str = Field(description="Clinical risk level ('Low', 'Moderate', 'High', 'Critical')")


class PredictionResponse(BaseModel):
    """Complete diagnostic prediction response payload."""
    top_prediction: ClassPrediction
    predictions: List[ClassPrediction]
    high_risk_alert: bool = Field(description="True if predicted lesion is malignant or precancerous (mel, bcc, akiec)")
    gradcam_base64: Optional[str] = Field(default=None, description="Base64 encoded Grad-CAM overlay image")
    inference_time_ms: float = Field(description="Time elapsed during inference in milliseconds")


class HealthResponse(BaseModel):
    """System health check and diagnostic status."""
    status: str
    version: str
    model_loaded: bool
    active_backbone: str
    hardware_acceleration: str
