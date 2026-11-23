# DermaAI API Contract & Security Specification

## Core Endpoints
- `POST /api/v1/predict`: Single-lesion prediction with Grad-CAM heatmap generation.
- `POST /api/v1/batch-predict`: Multi-image batch inference queue.
- `GET /health`: Readiness and liveness probe reporting model status and memory load.

## Security Layers
- **HMAC Signatures:** Prevents payload tampering in transit.
- **Rate Limiting:** Sliding-window per client IP.
- **CSP & Security Headers:** Enforces OWASP-compliant response headers.
