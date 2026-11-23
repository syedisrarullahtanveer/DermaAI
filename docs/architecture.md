## Project Architecture

```
DermaAI/
├─ backend/                 # FastAPI service
│   ├─ main.py              # API entrypoint
│   └─ service.py           # Model inference wrapper
├─ frontend/                # React + Vite UI
│   ├─ src/
│   │   ├─ components/      # UI widgets (UploadImage, PredictionResult, GradCamViewer)
│   │   ├─ pages/           # Static pages (ModelInfo, About, Disclaimer)
│   │   └─ App.tsx          # Routing and layout
│   └─ index.html
├─ src/                     # Core ML / XAI utilities used by backend
│   ├─ models/              # Model definitions
│   ├─ explainability/      # Grad‑CAM implementation
│   └─ preprocessing/       # Image cleaning utilities
├─ tests/                   # Unit, integration and performance tests
├─ docs/                    # Documentation (markdown)
└─ .github/workflows/ci.yml # CI pipeline
```

The backend exposes a `/api/predict` endpoint that receives an image, runs preprocessing, performs inference and optionally returns a Grad‑CAM heatmap. The frontend consumes this API and visualises the results.
