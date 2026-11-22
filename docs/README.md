# DermaAI: Engineering and Research Documentation

Welcome to the comprehensive documentation suite for **DermaAI – Intelligent Skin Disease Detection**.

---

## 📚 Documentation Index

1. **[System Architecture](architecture.md)**:
   - Complete technical blueprint of data ingestion, preprocessing layers, deep learning backbones, attention mechanisms, multimodal fusion, and FastAPI serving.
2. **[Research Methodology](research_methodology.md)**:
   - Deep dive into clinical justification for DullRazor hair removal, class imbalance mitigation via Focal Loss, attention mechanism localization, and patient-aware stratification.
3. **[API Reference](api_reference.md)**:
   - Complete REST API contract, Pydantic data schemas, request/response examples, and error codes.

---

## 📂 Directory Organization Reference

| Directory | Purpose | Key Artifacts |
| :--- | :--- | :--- |
| `dataset/` | Data storage specifications and partition manifests | `metadata/`, `splits/`, documentation |
| `configs/` | Centralized hyperparameter and model configurations | `config.yaml`, `model_configs.yaml` |
| `notebooks/` | Interactive scientific exploration and benchmarks | Exploratory Data Analysis & XAI notebooks |
| `src/data/` | Data discovery, metadata parsing, and `tf.data` pipelines | `dataset_loader.py`, `metadata_parser.py` |
| `src/preprocessing/` | Artifact removal and augmentation algorithms | `image_cleaner.py` (DullRazor), `augmentations.py` |
| `src/models/` | Deep learning models and attention mechanisms | Baseline CNN, Transfer learning, CBAM, Multimodal |
| `src/training/` | Training loop orchestration and loss functions | `losses.py` (Focal Loss), `trainer.py`, `callbacks.py` |
| `src/evaluation/` | Multi-class clinical diagnostics and reports | `metrics.py`, `evaluator.py`, confusion matrices |
| `src/explainability/` | Visual localization and explainable AI | `gradcam.py` (Grad-CAM/Grad-CAM++), `visualizer.py` |
| `backend/` | High-throughput asynchronous REST API | FastAPI application, Pydantic schemas, inference engine |
| `frontend/` | Clinical decision support web dashboard | React + Vite UI, interactive Grad-CAM heatmap viewer |
| `tests/` | Unit and integration verification suite | Pytest suites for algorithms, models, and endpoints |
| `scripts/` | Command-line training and evaluation tools | `train.py`, `evaluate.py`, `verify_environment.py` |
| `.github/` | Continuous integration workflows | GitHub Actions CI (`ci.yml`) |
