"""
Explanation faithfulness evaluation via pixel insertion and deletion curves.
"""
import numpy as np

class FaithfulnessEvaluator:
    def compute_deletion_score(self, predict_fn, image: np.ndarray, saliency_map: np.ndarray, steps: int = 5) -> float:
        flat_saliency = saliency_map.ravel()
        sorted_indices = np.argsort(flat_saliency)[::-1] # highest saliency first
        img = image.copy()
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim == 3 else 1
        
        step_size = len(sorted_indices) // steps
        preds = []
        for i in range(steps):
            to_mask = sorted_indices[: (i + 1) * step_size]
            coords = np.unravel_index(to_mask, (h, w))
            img[coords] = 0
            score = float(predict_fn(img))
            preds.append(score)
            
        return float(np.mean(preds))
