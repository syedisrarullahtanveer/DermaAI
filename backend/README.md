# DermaAI: Production Backend REST API

The backend is built with **FastAPI** to provide high-performance, asynchronous dermatological diagnosis inference, image preprocessing with DullRazor hair removal, and Explainable AI (Grad-CAM) heatmap generation.

---

## ⚡ API Endpoints

### 1. `GET /api/health`
Health check endpoint reporting framework status, model loaded state, active backbone, and hardware acceleration (GPU/CPU).

**Example Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "active_backbone": "efficientnet_b0",
  "hardware_acceleration": "CPU"
}
```

### 2. `POST /api/predict`
Uploads a dermoscopy image (`multipart/form-data`) and returns:
- `top_prediction`: Highest probability diagnosis with clinical risk level.
- `predictions`: Ranked list of all 7 lesion classes.
- `high_risk_alert`: Boolean flag indicating malignant or precancerous status (`mel`, `bcc`, `akiec`).
- `gradcam_base64`: Base64 encoded JPEG of the Grad-CAM thermal overlay on the lesion.
- `inference_time_ms`: Execution latency in milliseconds.

**Query Parameters:**
- `apply_hair_removal`: (bool, default `true`) - Toggles DullRazor morphological filter.
- `include_gradcam`: (bool, default `true`) - Toggles Grad-CAM heatmap generation.

---

## 🚀 Running the API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
