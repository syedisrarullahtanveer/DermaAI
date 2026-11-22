# Publicly Available Skin Lesion Datasets & Licensing Registry

This registry provides detailed metadata, research citations, and strict licensing terms for publicly available dermoscopy and clinical skin lesion datasets.

---

## 1. HAM10000 Dataset ("Human Against Machine with 10,000 Training Images")

| Attribute | Details |
| :--- | :--- |
| **Origin / Providers** | Department of Dermatology, Medical University of Vienna, Austria & Cliff Rosendahl Clinic, Queensland, Australia |
| **Release Year** | 2018 |
| **Modality** | Dermatoscopic high-resolution images (epiluminescence microscopy) |
| **Total Samples** | 10,015 images representing 7,470 unique lesions |
| **Class Taxonomy** | 7 classes: `mel` (Melanoma), `nv` (Melanocytic Nevi), `bcc` (Basal Cell Carcinoma), `akiec` (Actinic Keratoses), `bkl` (Benign Keratosis), `df` (Dermatofibroma), `vasc` (Vascular lesions) |
| **Ground Truth Validation** | Histopathology (~53.3%), Confocal microscopy, Follow-up consensus |
| **License** | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** |
| **Usage Restrictions** | Free for academic, scientific, and non-commercial research. Commercial redistribution or monetized deployment requires explicit licensing. |
| **Official Source** | Harvard Dataverse: `https://doi.org/10.7910/DVN/DBW86T` |
| **ISIC Challenge** | ISIC 2018 Challenge Task 3 Benchmark |

---

## 2. ISIC (International Skin Imaging Collaboration) Archive

| Benchmark | Total Images | Classes | Modality | License | Key Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ISIC 2018** | 10,015 (Train) | 7 | Dermoscopy | CC BY-NC 4.0 | Based on HAM10000 |
| **ISIC 2019** | 25,331 | 8 | Dermoscopy | CC BY-NC 4.0 | Incorporates BCN20000, added Squamous Cell Carcinoma (`scc`) |
| **ISIC 2020** | 33,126 | Binary (Malignant vs. Benign) | Dermoscopy + Clinical | CC BY-NC 4.0 | Focused on melanoma detection in context of patient lesion profiles |
| **ISIC 2024** | 400,000+ | Multiple | 3D TBP (Total Body Photography) & Tile Crops | CC-BY 4.0 | Evaluates lesions cropped from 3D total body photography |

* **Website**: `https://www.isic-archive.com/`

---

## 3. PAD-UFES-20 Dataset

| Attribute | Details |
| :--- | :--- |
| **Origin / Providers** | Federal University of Espírito Santo (UFES), Brazil |
| **Release Year** | 2020 |
| **Modality** | Clinical smartphone camera photography (non-dermoscopic macro images) |
| **Total Samples** | 2,298 images from 1,373 unique patients |
| **Classes** | 6 classes: Actinic Keratosis (ACK), Basal Cell Carcinoma (BCC), Melanoma (MEL), Nevus (NEV), Seborrheic Keratosis (SEK), Squamous Cell Carcinoma (SCC) |
| **Metadata** | 21 patient clinical demographic features (age, sex, skin tone Fitzpatrick scale, itch, bleeding, elevation, growth, smoke, drink, pesticide exposure) |
| **License** | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| **Commercial Use** | Permitted under CC BY 4.0 with appropriate attribution |
| **Source** | Mendeley Data: `https://doi.org/10.17632/zr7vgbcyr2.1` |

---

## 4. BCN20000 Dataset

| Attribute | Details |
| :--- | :--- |
| **Origin / Providers** | Hospital Clínic de Barcelona, Spain |
| **Release Year** | 2019 |
| **Modality** | Dermoscopy of hard-to-diagnose, difficult anatomical sites (mucosa, nails, acral) |
| **Total Samples** | 19,424 images |
| **License** | **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)** |
| **Source** | ISIC Archive / Zenodo `https://zenodo.org/record/3404368` |

---

## 5. Compliance & Ethical Usage Guidelines

1. **Patient Anonymity**: All datasets mentioned above have been de-identified and stripped of protected health information (PHI) in compliance with HIPAA and GDPR research exemptions.
2. **Attribution**: Whenever publishing research, training models, or presenting benchmarks, proper academic citations must be given to the dataset authors (e.g., Tschandl et al., *Nature Scientific Data* 2018 for HAM10000).
3. **Storage Security**: Raw patient images must remain in `dataset/raw/` and must never be uploaded or committed to public version control repositories.
