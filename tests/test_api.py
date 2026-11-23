"""Integration Tests for FastAPI Backend Endpoints."""

import io
import unittest
from PIL import Image

try:
    from fastapi.testclient import TestClient
    import pytest
    from backend.main import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class TestAPIEndpoints(unittest.TestCase):
    @unittest.skipUnless(HAS_FASTAPI, "FastAPI or dependencies not installed")
    def test_health_check_endpoint(self):
        """Verify /api/health returns 200 OK and valid diagnostic payload."""
        client = TestClient(app)
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "1.0.0")
        self.assertIn("model_loaded", data)

    @unittest.skipUnless(HAS_FASTAPI, "FastAPI or dependencies not installed")
    def test_predict_endpoint_with_image(self):
        """Verify /api/predict accepts dermoscopic image and returns top prediction and 7 classes."""
        client = TestClient(app)
        img = Image.new("RGB", (224, 224), color=(180, 120, 110))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        response = client.post(
            "/api/predict",
            files={"file": ("test_lesion.jpg", img_byte_arr, "image/jpeg")},
            params={"apply_hair_removal": False, "include_gradcam": True},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("top_prediction", data)
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 7)
        self.assertIn("high_risk_alert", data)
        self.assertIn("inference_time_ms", data)
        self.assertGreater(data["top_prediction"]["probability"], 0.0)

    @unittest.skipUnless(HAS_FASTAPI, "FastAPI or dependencies not installed")
    def test_predict_endpoint_rejects_invalid_file(self):
        """Verify /api/predict rejects non-image file uploads."""
        client = TestClient(app)
        response = client.post(
            "/api/predict",
            files={"file": ("notes.txt", b"plain text data", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
