"""FastAPI REST API Application Entrypoint for DermaAI."""

import os
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import PredictionResponse, HealthResponse, ClinicalMetadataInput
from backend.service import ModelInferenceService
from src.utils import setup_logger, get_device_info

logger = setup_logger("DermaAI.API")

app = FastAPI(
    title="DermaAI – Intelligent Skin Disease Detection API",
    description="Research-oriented dermatological lesion classification API featuring Attention Mechanisms and Explainable AI (Grad-CAM).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
# Add Request ID middleware
from backend.middleware.request_id import RequestIdMiddleware
app.add_middleware(RequestIdMiddleware)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Warm up deep learning model during application startup."""
    logger.info("Initializing DermaAI Backend API service...")
    try:
        _ = ModelInferenceService.get_instance()
        logger.info("DermaAI inference service successfully preloaded.")
    except Exception as e:
        logger.error(f"Failed to preload model service: {e}")


@app.get("/api/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Verify backend health, framework versions, and model availability."""
    service = ModelInferenceService.get_instance()
    device_info = get_device_info()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=(service.model is not None),
        active_backbone=service.backbone_name,
        hardware_acceleration="GPU" if device_info.get("gpu_available") else "CPU",
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_lesion(
    file: UploadFile = File(..., description="Dermoscopic image file (JPEG or PNG)"),
    apply_hair_removal: bool = Query(True, description="Apply DullRazor hair removal filter"),
    include_gradcam: bool = Query(True, description="Generate and return Grad-CAM visual explanation"),
):
    """Classify uploaded skin lesion image and return top-k diagnosis probabilities and Grad-CAM explanation."""
    # Validate MIME type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Supported formats are JPEG and PNG.",
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        service = ModelInferenceService.get_instance()
        result = service.predict(
            image_bytes=image_bytes,
            include_gradcam=include_gradcam,
            apply_dull_razor=apply_hair_removal,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference pipeline failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
