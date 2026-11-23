"""
Encodes clinical metadata (age, sex, anatomical site) into standardized numerical vectors.
"""
import numpy as np

class MetadataFeatureEncoder:
    SITES = ["head/neck", "upper extremity", "lower extremity", "torso", "palms/soles", "oral/genital", "unknown"]
    SEXES = ["male", "female", "unknown"]

    def encode(self, age: float, sex: str, site: str) -> np.ndarray:
        # Normalized age: [0, 100] -> [0.0, 1.0]
        norm_age = min(1.0, max(0.0, float(age) / 100.0))
        
        # One-hot sex
        sex_lower = (sex or "unknown").lower()
        sex_vec = [1.0 if sex_lower == s else 0.0 for s in self.SEXES]
        if sum(sex_vec) == 0: sex_vec[-1] = 1.0
        
        # One-hot site
        site_lower = (site or "unknown").lower()
        site_vec = [1.0 if site_lower == st else 0.0 for st in self.SITES]
        if sum(site_vec) == 0: site_vec[-1] = 1.0
        
        return np.array([norm_age] + sex_vec + site_vec, dtype=np.float32)
