# Dermoscopy Image Preprocessing and Calibration Pipeline

## Pipeline Sequence
1. **EXIF Sanitization:** Strips all metadata to protect patient PHI.
2. **Quality Verification:** Evaluates Laplacian variance (>100) and JPEG blockiness.
3. **Artifact Removal:** Applies DullRazor hair mask filtering and vignette attenuation.
4. **Color Constancy:** Normalizes illumination using Shades of Gray / Gray World.
5. **Letterbox Padding:** Preserves aspect ratio when scaling to 224x224.
