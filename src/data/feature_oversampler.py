"""
SMOTE feature-space linear interpolation oversampler for under-represented skin lesion classes.
"""
import numpy as np

class FeatureSpaceOversampler:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def resample_class(self, features: np.ndarray, target_count: int) -> np.ndarray:
        current_count = len(features)
        if current_count >= target_count or current_count < 2:
            return features
        
        diff = target_count - current_count
        synthetic = []
        for _ in range(diff):
            i1, i2 = self.rng.choice(current_count, size=2, replace=False)
            lam = self.rng.uniform(0.0, 1.0)
            syn = features[i1] + lam * (features[i2] - features[i1])
            synthetic.append(syn)
            
        return np.vstack([features, np.array(synthetic)])
