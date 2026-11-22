"""Class Imbalance Mitigation Strategies for Dermatological Datasets."""

from typing import Dict, List, Optional, Union
import numpy as np


class ClassImbalanceHandler:
    """Calculates class weights, effective number weighting, and minority augmentation multipliers."""

    @staticmethod
    def compute_balanced_weights(class_counts: Dict[Union[str, int], int]) -> Dict[Union[str, int], float]:
        """Compute standard balanced heuristic weights: w_c = N / (K * N_c).

        Args:
            class_counts: Mapping from class label to sample count.

        Returns:
            Dict: Normalized class weights where mean weight is 1.0.
        """
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        weights = {}

        for cls, count in class_counts.items():
            safe_count = max(count, 1)
            weights[cls] = float(total_samples / (num_classes * safe_count))

        return weights

    @staticmethod
    def compute_effective_number_weights(
        class_counts: Dict[Union[str, int], int],
        beta: float = 0.999,
    ) -> Dict[Union[str, int], float]:
        """Compute Class-Balanced Loss weights based on the Effective Number of Samples (Cui et al., CVPR 2019).

        E_n = (1 - beta^n) / (1 - beta)
        weight_c = 1 / E_{n_c}

        Args:
            class_counts: Mapping from class label to sample count.
            beta: Hyperparameter in range [0.9, 0.9999]. As beta -> 1, approaches inverse frequency.

        Returns:
            Dict: Normalized class weights where sum(weights) = num_classes.
        """
        assert 0.0 < beta < 1.0, "Beta must be strictly between 0 and 1."
        effective_num = {}
        for cls, count in class_counts.items():
            safe_count = max(count, 1)
            effective_num[cls] = (1.0 - (beta ** safe_count)) / (1.0 - beta)

        raw_weights = {cls: 1.0 / max(eff, 1e-6) for cls, eff in effective_num.items()}

        # Normalize weights so that their sum equals the number of classes
        total_raw = sum(raw_weights.values())
        k = len(class_counts)
        normalized = {cls: float((w / total_raw) * k) for cls, w in raw_weights.items()}
        return normalized

    @staticmethod
    def compute_augmentation_multipliers(
        class_counts: Dict[Union[str, int], int],
        target_fraction_of_max: float = 0.40,
        max_multiplier: int = 10,
    ) -> Dict[Union[str, int], int]:
        """Compute integer augmentation factors for minority classes to level the training playing field.

        Args:
            class_counts: Mapping from class label to sample count.
            target_fraction_of_max: Desired target sample count as fraction of majority class.
            max_multiplier: Upper cap on augmentation copies to prevent extreme duplication.

        Returns:
            Dict: Integer multiplier per class (1 = no extra augmentations, >1 = generate extra copies).
        """
        max_count = max(class_counts.values()) if class_counts else 1
        target_count = int(max_count * target_fraction_of_max)

        multipliers = {}
        for cls, count in class_counts.items():
            if count >= target_count:
                multipliers[cls] = 1
            else:
                multiplier = int(np.ceil(target_count / max(count, 1)))
                multipliers[cls] = min(multiplier, max_multiplier)

        return multipliers

    @staticmethod
    def compute_sample_weights(
        labels: np.ndarray,
        class_weights: Dict[Union[str, int], float],
    ) -> np.ndarray:
        """Map sample labels to their corresponding class weight values."""
        sample_weights = np.array([class_weights.get(lbl, 1.0) for lbl in labels], dtype=np.float32)
        return sample_weights
