# DermaAI

**DermaAI – Intelligent Skin Disease Detection**

DermaAI is an open‑source research prototype that combines deep learning, explainable AI (Grad‑CAM), and a modern web dashboard to assist clinicians and researchers in analyzing dermoscopic images.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Methodology](#model-methodology)
- [Experimental Results](#experimental-results)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Limitations & Ethical Considerations](#limitations--ethical-considerations)
- [Deployment](#deployment)
- [Development & Testing](#development--testing)
- [Contributing](#contributing)
- [License](#license)

## Overview
DermaAI provides a **FastAPI** backend that:
- Accepts dermoscopic image uploads.
- Performs pre‑processing (hair removal, color constancy).
- Runs an EfficientNet‑B0 backbone with CBAM/SE attention modules.
- Returns top‑k predictions, confidence scores, and a Grad‑CAM heatmap.

The **React + Vite** frontend offers a responsive medical‑style dashboard (blue‑and‑white theme) with:
- Image upload & preview.
- Real‑time inference results.
- Class‑wise probability bars.
- Grad‑CAM visualisation.
- Model information, about page, and disclaimer.

> **⚠️ Disclaimer:** DermaAI is a research prototype and **does not provide medical diagnoses**. Results are for educational and experimental purposes only.

## Features
- End‑to‑end inference pipeline with explainability.
- Robust API with validation and error handling.
- Automated CI pipeline (GitHub Actions) running unit, integration and performance tests.
- Full documentation and reproducible experiment reports.

## Installation
See the full [Installation Guide](docs/installation.md) for step‑by‑step instructions, but the quick‑start is:
```bash
# Clone the repo
git clone https://github.com/your-org/DermaAI.git
cd DermaAI

# Backend (Python)
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn backend.main:app --reload

# Frontend (Node)
cd frontend
npm install  # or pnpm install (requires network)
npm run dev
```

## Dataset
DermaAI was trained on the **ISIC 2018** dataset (7 diagnostic classes). Details on data acquisition, preprocessing, and licensing are documented in [Dataset Documentation](docs/dataset.md).

## Model Methodology
The model architecture, training schedule, loss functions and explainability pipeline are described in [Model Methodology](docs/methodology.md).

## Experimental Results
Benchmark results, confusion matrices and class‑wise ROC curves are summarised in [Experimental Results](docs/results.md).

## API Documentation
The FastAPI OpenAPI schema is available at `http://localhost:8000/docs`. A concise reference is kept in [API Documentation](docs/api.md).

## Screenshots
![Dashboard](docs/screenshots/dashboard.png)
![Grad‑CAM overlay](docs/screenshots/gradcam.png)

## Limitations & Ethical Considerations
DermaAI is trained on a limited public dataset and may not generalise to all skin tones or rare conditions. See [Limitations & Ethics](docs/limitations.md) for a thorough discussion.

## Deployment
Guidelines for containerising and deploying to cloud platforms (Docker, Kubernetes, GCP) are in [Deployment Instructions](docs/deployment.md).

## Development & Testing
- Run all tests with `pytest -q`.
- Integration tests cover the full API‑frontend flow.
- CI runs on every push (see `.github/workflows/ci.yml`).

## Contributing
Please read the [Contribution Guide](CONTRIBUTING.md) before submitting pull requests.

## License
DermaAI is released under the MIT License. See `LICENSE` for details.
