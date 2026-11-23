"""
Stratified K-fold cross-validation split generator for multiclass dermoscopy.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

class StratifiedKFoldSplitter:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: int = 42):
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    def split(self, x: np.ndarray, y: np.ndarray):
        y_labels = np.asarray(y)
        if y_labels.ndim > 1:
            y_labels = np.argmax(y_labels, axis=-1)
        return list(self.skf.split(x, y_labels))
