# DermaAI: Dataset Directory Structure & Acquisition Guide

This directory manages all dermatological image data and clinical metadata.

---

## 📁 Directory Layout

```
dataset/
├── raw/                # Original downloaded dermoscopy images (ISIC / HAM10000)
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
├── processed/          # Cleaned, hair-removed, and normalized images
│   ├── train/
│   ├── val/
│   └── test/
├── metadata/           # Clinical patient metadata tables
│   └── HAM10000_metadata.csv
└── splits/             # Stratified patient-aware partition manifests
    ├── train.csv
    ├── val.csv
    └── test.csv
```

> **Note on Version Control:** Large raw image files and model checkpoints are strictly excluded from GitHub via `.gitignore`. Only directory `.gitkeep` files, schema documentation, and small sample metadata are tracked in Git.

---

## 🔬 Dataset Benchmark: HAM10000

The primary benchmark for DermaAI is the **HAM10000 ("Human Against Machine with 10,000 training images")** dataset, collected by the Department of Dermatology at the Medical University of Vienna and Cliff Rosendahl in Queensland, Australia.

### Disease Classes (7 Lesion Types)

| Code | Diagnosis | Lesion Category | Clinical Risk |
| :--- | :--- | :--- | :--- |
| **`mel`** | Melanoma | Malignant neoplasm of melanocytes | **Critical (Life-threatening)** |
| **`bcc`** | Basal Cell Carcinoma | Malignant epithelial skin cancer | **High** |
| **`akiec`** | Actinic Keratosis / Bowen's Disease | Precancerous / Intraepithelial carcinoma | **High** |
| **`bkl`** | Benign Keratosis | Seborrheic keratosis, solar lentigo, lichen-planus | **Low** |
| **`df`** | Dermatofibroma | Benign dermal histiocytoma | **Low** |
| **`vasc`** | Vascular Lesions | Cherry angiomas, angiokeratomas, pyogenic granulomas | **Low** |
| **`nv`** | Melanocytic Nevi | Common benign moles | **Low (Dominant class)** |

---

## 📊 Metadata Schema

The metadata file (`dataset/metadata/HAM10000_metadata.csv`) contains:
- `lesion_id`: Unique identifier for each lesion. Multiple dermoscopic images can belong to the same lesion or patient.
- `image_id`: Unique identifier for each image (e.g. `ISIC_0027419`).
- `dx`: Ground-truth diagnosis (`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`).
- `dx_type`: Ground truth confirmation method (`histo` for histopathology, `follow_up`, `consensus`, `confocal`).
- `age`: Patient age in years (numerical).
- `sex`: Patient biological sex (`male`, `female`, `unknown`).
- `localization`: Anatomical site where the lesion appeared (`back`, `lower extremity`, `trunk`, `upper extremity`, `abdomen`, `face`, `chest`, `neck`, etc.).

---

## ⚠️ Patient Overlap & Data Leakage Prevention

A single `lesion_id` often has multiple images taken at different times or angles.
**Crucial Medical Rule:** Train, validation, and test splits **must be grouped by `lesion_id`**, ensuring all images of a particular lesion exist exclusively in one split. DermaAI's `src/data/dataset_loader.py` enforces this patient-aware stratified grouping automatically.
