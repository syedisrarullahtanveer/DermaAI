# DermaAI: Clinical Frontend Dashboard

The frontend is a clinical diagnostic interface built with **React** and styled for clean, medical-grade decision support.

---

## 🖥️ Features

- **Drag-and-Drop Image Ingestion**: Upload dermoscopy scans with instant preview.
- **Clinical Parameter Controls**: Toggle DullRazor hair removal and Grad-CAM generation.
- **Patient Metadata Input**: Enter optional age, biological sex, and lesion anatomical site.
- **Dynamic Diagnostic Breakdown**:
  - Critical/High-Risk malignant alert banners for Melanoma (`mel`), Basal Cell Carcinoma (`bcc`), and Actinic Keratosis (`akiec`).
  - Real-time confidence percentage bar charts across all 7 lesion categories.
- **Explainable AI Visualizer**:
  - Interactive comparison toggle between the original dermoscopy image and the Grad-CAM thermal activation heatmap.
  - Explains lesion perimeter and pigment network focus areas.

---

## 🚀 Running Frontend Locally

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at: `http://localhost:5173`.
