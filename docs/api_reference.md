# DermaAI: REST API Reference Specification

---

## Base URL
```
http://localhost:8000
```

---

## Endpoints

### 1. Health & Diagnostics Check
- **Route**: `GET /api/health`
- **Description**: Verifies backend availability, framework runtime, and hardware acceleration status.
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "active_backbone": "efficientnet_b0",
  "hardware_acceleration": "CPU"
}
```

---

### 2. Lesion Classification & Grad-CAM Inference
- **Route**: `POST /api/predict`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Binary dermoscopic image file (JPEG or PNG) [Required]
  - `apply_hair_removal`: (bool, query parameter, default `true`) - Toggles DullRazor filter.
  - `include_gradcam`: (bool, query parameter, default `true`) - Toggles Grad-CAM thermal overlay.
- **Response**: `200 OK`
```json
{
  "top_prediction": {
    "code": "mel",
    "name": "Melanoma",
    "probability": 0.8842,
    "risk_level": "Critical"
  },
  "predictions": [
    {
      "code": "mel",
      "name": "Melanoma",
      "probability": 0.8842,
      "risk_level": "Critical"
    },
    {
      "code": "nv",
      "name": "Melanocytic Nevi",
      "probability": 0.0712,
      "risk_level": "Low"
    },
    {
      "code": "bcc",
      "name": "Basal Cell Carcinoma",
      "probability": 0.0241,
      "risk_level": "High"
    },
    {
      "code": "bkl",
      "name": "Benign Keratosis-like Lesions",
      "probability": 0.0120,
      "risk_level": "Low"
    },
    {
      "code": "akiec",
      "name": "Actinic Keratoses and Intraepithelial Carcinoma",
      "probability": 0.0051,
      "risk_level": "High"
    },
    {
      "code": "df",
      "name": "Dermatofibroma",
      "probability": 0.0022,
      "risk_level": "Low"
    },
    {
      "code": "vasc",
      "name": "Vascular Lesions",
      "probability": 0.0012,
      "risk_level": "Low"
    }
  ],
  "high_risk_alert": true,
  "gradcam_base64": "/9j/4AAQSkZJRgABAQ...",
  "inference_time_ms": 112.45
}
```
