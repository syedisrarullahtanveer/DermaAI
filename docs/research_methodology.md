# DermaAI: Research Methodology and Clinical Rationale

---

## 1. Dermatological Problem Statement

Skin cancer is among the most commonly diagnosed cancers globally. Early and accurate detection of **Melanoma (`mel`)**, the deadliest form of skin cancer, dramatically increases 5-year survival rates from below 25% to over 98%. However, clinical visual diagnosis is confounded by visual similarities between malignant melanoma and benign melanocytic nevi (`nv`) or seborrheic keratoses (`bkl`).

DermaAI addresses this challenge through automated multi-class classification over the established **HAM10000** 7-class taxonomy.

---

## 2. Resolving Class Imbalance with Focal Loss

The HAM10000 benchmark exhibits extreme class imbalance:
- **Melanocytic Nevi (`nv`)**: ~67% of all samples
- **Dermatofibroma (`df`)**: ~1.1% of all samples
- **Vascular Lesions (`vasc`)**: ~1.4% of all samples

Standard cross-entropy loss allows the massive majority class (`nv`) to dominate gradient updates, leading to catastrophic false-negative rates on rare but critical classes like Melanoma (`mel`).

DermaAI implements **Multi-Class Focal Loss**:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Where:
- $\gamma = 2.0$: Suppresses gradients from well-classified easy examples.
- $\alpha_t$: Inversely scales loss according to class rarity.

---

## 3. Patient-Aware Stratified Partitioning

Many clinical benchmarks suffer from **data leakage** when multiple images of the same patient lesion are scattered across both training and testing sets.
DermaAI eliminates this artifact by strictly enforcing **Grouped Stratified Partitioning by `lesion_id`**:
- All images sharing a `lesion_id` remain exclusively in one partition (Train, Validation, or Test).
- Partition distributions are simultaneously stratified by diagnostic class (`dx`).

---

## 4. Explainable AI for Clinical Adoption

Black-box neural networks face significant skepticism from medical practitioners. DermaAI integrates **Grad-CAM (Gradient-Weighted Class Activation Mapping)** to visualize the neural activation gradient over the lesion:
- Confirms the model attends to authentic biological features (e.g. pigment network asymmetry, irregular border structures) rather than imaging artifacts (such as skin markers, gel bubbles, or dark corner vignettes).
