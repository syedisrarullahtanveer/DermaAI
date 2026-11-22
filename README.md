# DermaAI: Intelligent Skin Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Status-Active%20Development-success.svg)]()

> **DermaAI** is an advanced, research-oriented deep learning system engineered for multi-class skin lesion classification and clinical diagnostic decision support. It combines modern computer vision architectures (ResNet, EfficientNet, MobileNet), Convolutional Block Attention Modules (CBAM), multimodal tabular metadata fusion, and Explainable AI (Grad-CAM/Grad-CAM++) into a production-ready web platform.

---

## 🌟 Key Highlights & Architectural Features

- **7-Class Lesion Taxonomy (HAM10000 Benchmark)**:
  - `MEL`: Melanoma *(Critical)*
  - `BCC`: Basal Cell Carcinoma *(High)*
  - `AKIEC`: Actinic Keratoses and Intraepithelial Carcinoma *(High)*
  - `BKL`: Benign Keratosis-like Lesions *(Low)*
  - `DF`: Dermatofibroma *(Low)*
  - `VASC`: Vascular Lesions *(Low)*
  - `NV`: Melanocytic Nevi *(Low)*
- **Advanced Preprocessing Engine**:
  - **DullRazor Algorithm**: Morphological black-hat filtering and bi-cubic inpainting for complete artifact and hair occlusion elimination.
  - **Color Constancy**: Gray World illumination normalization to harmonize dermoscopic imaging variations across different lighting sources.
- **Deep Architecture Suite**:
  - Baseline modular Deep CNN with residual connections and spatial regularization.
  - Transfer learning backbones: **ResNet50**, **EfficientNetB0**, and **MobileNetV2**.
  - **CBAM Attention**: Channel and Spatial Attention mechanisms focusing on clinically distinctive lesion borders and pigment patterns.
  - **Multimodal Fusion**: Dual-branch architecture merging high-level visual representations with clinical tabular features (patient age, sex, anatomical site).
- **Explainable AI (XAI)**:
  - Real-time **Grad-CAM** and **Grad-CAM++** heatmap synthesis for clinical transparency and localized diagnostic justification.
- **Full-Stack Ecosystem**:
  - **FastAPI Backend**: Asynchronous inference pipeline, strict Pydantic input/output schemas, and OpenAPI documentation.
  - **Frontend Dashboard**: Medical-grade diagnostic UI with real-time probability charts, lesion zoom, and Grad-CAM opacity controls.
- **Robustness & Engineering**:
  - Patient-aware stratified dataset partitioning preventing patient overlap leakage.
  - Focal loss and class-weighted cross-entropy addressing severe class imbalance.

---

## 📂 Project Architecture

```
DermaAI/
├── dataset/                  # Dataset specifications and storage guides
│   ├── raw/                  # Raw HAM10000/ISIC dermoscopy images (git-ignored)
│   ├── processed/            # Cleaned, hair-removed, normalized images (git-ignored)
│   └── metadata/             # Clinical metadata CSV tables
├── configs/                  # Centralized YAML hyperparameter and model configs
├── notebooks/                # Exploratory Data Analysis and research prototyping
├── src/                      # Production Python package
│   ├── data/                 # Data loaders, patient splitting, tf.data pipelines
│   ├── preprocessing/        # DullRazor, color constancy, Albumentations
│   ├── models/               # Baseline CNN, Transfer suite, CBAM Attention, Fusion
│   ├── training/             # Loss functions (Focal Loss), trainer, callbacks
│   ├── evaluation/           # Multiclass metrics, ROC curves, Confusion Matrix
│   ├── explainability/       # Grad-CAM and Grad-CAM++ localization engines
│   └── utils.py              # System logger, reproducibility seeds, device utils
├── backend/                  # FastAPI inference REST API service
├── frontend/                 # Interactive medical-grade clinical dashboard
├── tests/                    # Pytest test suite for data, models, XAI, and API
├── docs/                     # Architectural, research, and API documentation
└── scripts/                  # CLI utilities for verification, training, and evaluation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/syedisrarullahtanveer/DermaAI.git
cd DermaAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Environment & Hardware

```bash
python scripts/verify_environment.py
```

### 3. Model Training & Evaluation

```bash
# Train baseline CNN
python scripts/train.py --model baseline_cnn

# Train Attention-augmented network
python scripts/train.py --model attention_cbam

# Evaluate model performance
python scripts/evaluate.py --model attention_cbam --weights models/saved_models/best_model.keras
```

### 4. Run Backend API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at: `http://localhost:8000/docs`.

---

## 📈 Development Progress

Development follows a strict **400-Commit Incremental Roadmap**.
Detailed commit logs, phase completion metrics, and task statuses are tracked in [`PROGRESS.md`](PROGRESS.md).

---

## 📄 License & Attribution

This project is licensed under the MIT License.
Dataset acknowledgments: The International Skin Imaging Collaboration (ISIC) and the HAM10000 dataset (*ViDIR Group, Medical University of Vienna*).
