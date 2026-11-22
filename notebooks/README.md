# DermaAI: Research Notebooks

This folder contains Jupyter notebooks used for experimental analysis, data exploration, and algorithmic benchmarking.

---

## 📓 Notebook Index

1. **`01_eda_and_metadata_analysis.ipynb`**:
   - Comprehensive Exploratory Data Analysis (EDA) of the HAM10000 dataset.
   - Analysis of severe class imbalance, patient lesion frequency, age distribution, anatomical localization correlations.
2. **`02_preprocessing_and_artifacts.ipynb`**:
   - DullRazor morphological hair removal filter evaluation.
   - Illumination normalization using Gray World color constancy.
   - Albumentations visual pipeline demonstrations.
3. **`03_baseline_cnn_experiments.ipynb`**:
   - Training modular custom CNN backbones from scratch.
   - Benchmarking Focal Loss vs. Categorical Cross-Entropy with label smoothing.
4. **`04_transfer_learning_benchmarks.ipynb`**:
   - Comparative transfer learning analysis: ResNet50 vs. EfficientNetB0 vs. MobileNetV2.
   - Staged fine-tuning and differential learning rate strategies.
5. **`05_attention_mechanisms_xai.ipynb`**:
   - Convolutional Block Attention Module (CBAM) feature map visualizations.
   - Explainable AI: Grad-CAM and Grad-CAM++ lesion localization.
6. **`06_multimodal_fusion_evaluation.ipynb`**:
   - Integrating dermoscopy images with patient clinical tabular features (age, sex, site).
   - Ablation analysis: Image-only vs. Metadata-only vs. Gated Multimodal Fusion.
