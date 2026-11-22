"""Integration Tests for FastAPI Backend Endpoints."""

import io
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify /api/health returns 200 OK and valid diagnostic payload."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "model_loaded" in data


def test_predict_endpoint_with_image():
    """Verify /api/predict accepts dermoscopic image and returns top prediction and 7 classes."""
    # Generate in-memory synthetic RGB test image
    img = Image.new("RGB", (224, 224), color=(180, 120, 110))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/api/predict",
        files={"file": ("test_lesion.jpg", img_byte_arr, "image/jpeg")},
        params={"apply_hair_removal": False, "include_gradcam": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert "top_prediction" in data
    assert "predictions" in data
    assert len(data["predictions"]) == 7
    assert "high_risk_alert" in data
    assert "inference_time_ms" in data
    assert data["top_prediction"]["probability"] > 0.0


def test_predict_endpoint_rejects_invalid_file():
    """Verify /api/predict rejects non-image file uploads."""
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"plain text data", "text/plain")},
    )
    assert response.status_code == 400
