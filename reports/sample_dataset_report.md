# DermaAI: Clinical Dataset Audit & Distribution Report

## 1. Executive Summary
- **Total Dermoscopy Records:** 70
- **Unique Patient Lesions:** 28
- **Class Imbalance Ratio (IR):** **1.0 : 1** (Majority: `akiec`, Minority: `akiec`)
- **Normalized Class Entropy:** 1.0 / 1.000

## 2. Diagnosis Class Distribution

| Diagnosis Code | Clinical Category | Sample Count | Percentage |
| :--- | :--- | :--- | :--- |
| `akiec` | Actinic Keratosis (High) | 10 | 14.29% |
| `bcc` | Basal Cell Carcinoma (High) | 10 | 14.29% |
| `bkl` | Benign Keratosis (Low) | 10 | 14.29% |
| `df` | Dermatofibroma (Low) | 10 | 14.29% |
| `mel` | Melanoma (Critical) | 10 | 14.29% |
| `nv` | Melanocytic Nevus (Low) | 10 | 14.29% |
| `vasc` | Vascular Lesion (Low) | 10 | 14.29% |

## 3. Patient Lesion Multiplicity Audit
- **Single-Image Lesions:** 7 (25.0%)
- **Multi-Image Lesions:** 21 (75.0%)
- **Max Images for Single Lesion:** 3
> **Medical Integrity Assertion:** Stratified partitioning must strictly group by `lesion_id` to guarantee zero data leakage between train and test splits.

## 4. Patient Demographics Summary
- **Patient Age:** Mean 52.4 ± 16.2 years (Range: 25.0 - 75.0 yrs, Median: 52.0)
- **Missing Age Records:** 7
- **Biological Sex Distribution:** Female: 36, Male: 34

---
*Report generated automatically by DermaAI Dataset Reporter.*