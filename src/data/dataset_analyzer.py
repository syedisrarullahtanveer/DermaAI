"""Statistical Dataset Distribution and Demographic Analysis Module."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from src.utils import setup_logger

logger = setup_logger("DermaAI.DataAnalyzer")


class DatasetDistributionAnalyzer:
    """Computes comprehensive class distributions, imbalance ratios, patient multiplicities, and demographic correlations."""

    def __init__(self, metadata_df: pd.DataFrame, class_col: str = "dx", patient_col: str = "lesion_id"):
        self.df = metadata_df.copy()
        self.class_col = class_col
        self.patient_col = patient_col

    def compute_class_distribution(self) -> Dict[str, Union[Dict[str, int], float]]:
        """Compute sample counts, percentages, and imbalance ratio across diagnostic classes."""
        counts = self.df[self.class_col].value_counts().to_dict()
        total = len(self.df)
        percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}

        max_count = max(counts.values()) if counts else 0
        min_count = min(counts.values()) if counts else 1
        imbalance_ratio = round(float(max_count / max(min_count, 1)), 2)

        # Shannon Entropy as a measure of class balance (1.0 = perfectly uniform)
        probs = [v / total for v in counts.values()]
        entropy = -sum(p * np.log(p) for p in probs)
        max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
        normalized_entropy = round(float(entropy / max_entropy), 4)

        return {
            "total_samples": total,
            "counts": counts,
            "percentages": percentages,
            "imbalance_ratio": imbalance_ratio,
            "normalized_entropy": normalized_entropy,
            "majority_class": max(counts, key=counts.get) if counts else None,
            "minority_class": min(counts, key=counts.get) if counts else None,
        }

    def compute_patient_multiplicity(self) -> Dict[str, Union[int, float]]:
        """Compute lesion-to-patient multiplicity metrics to audit duplicate image risks."""
        if self.patient_col not in self.df.columns:
            return {"error": f"Patient grouping column '{self.patient_col}' not found"}

        images_per_patient = self.df.groupby(self.patient_col).size()
        total_unique = len(images_per_patient)
        multi_image = int((images_per_patient > 1).sum())

        return {
            "total_records": len(self.df),
            "unique_lesions": total_unique,
            "single_image_lesions": int((images_per_patient == 1).sum()),
            "multi_image_lesions": multi_image,
            "multi_image_percentage": round((multi_image / total_unique) * 100, 2) if total_unique else 0,
            "max_images_per_lesion": int(images_per_patient.max()) if total_unique else 0,
            "mean_images_per_lesion": round(float(images_per_patient.mean()), 3) if total_unique else 0,
        }

    def compute_demographic_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for age, sex, and anatomical localization."""
        res = {}

        # Age stats
        if "age" in self.df.columns:
            valid_age = self.df["age"].dropna().astype(float)
            res["age"] = {
                "available_samples": len(valid_age),
                "missing_samples": int(self.df["age"].isna().sum()),
                "mean": round(float(valid_age.mean()), 1) if len(valid_age) else None,
                "std": round(float(valid_age.std()), 1) if len(valid_age) else None,
                "median": float(valid_age.median()) if len(valid_age) else None,
                "min": float(valid_age.min()) if len(valid_age) else None,
                "max": float(valid_age.max()) if len(valid_age) else None,
            }

        # Sex stats
        if "sex" in self.df.columns:
            sex_counts = self.df["sex"].fillna("unknown").value_counts().to_dict()
            res["sex"] = sex_counts

        # Localization stats
        if "localization" in self.df.columns:
            loc_counts = self.df["localization"].fillna("unknown").value_counts().to_dict()
            res["localization"] = loc_counts

        return res

    def compute_cross_tabulations(self) -> Dict[str, Dict]:
        """Compute categorical cross-tabulations (Diagnosis vs Sex, Diagnosis vs Localization)."""
        tables = {}
        if "sex" in self.df.columns:
            ct_sex = pd.crosstab(self.df[self.class_col], self.df["sex"].fillna("unknown"))
            tables["dx_vs_sex"] = ct_sex.to_dict()

        if "localization" in self.df.columns:
            ct_loc = pd.crosstab(self.df[self.class_col], self.df["localization"].fillna("unknown"))
            tables["dx_vs_localization"] = ct_loc.to_dict()

        return tables

    def generate_full_analysis_dict(self) -> Dict:
        """Aggregate all analytical metrics into a unified dictionary."""
        return {
            "class_distribution": self.compute_class_distribution(),
            "patient_multiplicity": self.compute_patient_multiplicity(),
            "demographics": self.compute_demographic_stats(),
            "cross_tabulations": self.compute_cross_tabulations(),
        }
