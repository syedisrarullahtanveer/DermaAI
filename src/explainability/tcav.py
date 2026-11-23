"""
Testing with Concept Activation Vectors (TCAV) for dermoscopic clinical concepts (e.g. pigment network, streaks).
Reference: Kim et al., "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)", ICML 2018.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

class TCAVExtractor:
    def __init__(self):
        self.cav = None

    def fit_cav(self, concept_activations: np.ndarray, non_concept_activations: np.ndarray):
        x = np.vstack([concept_activations, non_concept_activations])
        y = np.array([1] * len(concept_activations) + [0] * len(non_concept_activations))
        clf = LogisticRegression(max_iter=200)
        clf.fit(x, y)
        # CAV is the normal vector to the decision hyperplane
        self.cav = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
        return self.cav

    def compute_tcav_score(self, model_gradients: np.ndarray) -> float:
        if self.cav is None:
            raise ValueError("CAV not fitted yet.")
        # Directional derivative: grad . cav
        directional_derivatives = np.dot(model_gradients, self.cav)
        score = np.mean(directional_derivatives > 0)
        return float(score)
