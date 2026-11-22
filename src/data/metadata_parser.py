"""Clinical Metadata Parser and Validator for Dermatological Records."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class ClinicalMetadataParser:
    """Parses, validates, and cleans clinical tabular metadata associated with dermoscopy records."""

    REQUIRED_COLUMNS = ["lesion_id", "image_id", "dx"]
    OPTIONAL_COLUMNS = ["dx_type", "age", "sex", "localization"]

    def __init__(self, metadata_path: Union[str, Path]):
        self.metadata_path = Path(metadata_path)
        self.df: Optional[pd.DataFrame] = None

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata CSV and validate required column schemas.

        Returns:
            pd.DataFrame: Cleaned metadata DataFrame.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)
        self._validate_schema()
        self._clean_records()
        return self.df

    def _validate_schema(self) -> None:
        """Ensure all required clinical columns are present in the dataset."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Metadata is missing required columns: {missing}")

    def _clean_records(self) -> None:
        """Handle missing demographic values and standardize categorical entries."""
        # Standardize strings
        self.df["dx"] = self.df["dx"].astype(str).str.strip().str.lower()
        if "sex" in self.df.columns:
            self.df["sex"] = self.df["sex"].fillna("unknown").astype(str).str.strip().str.lower()
        if "localization" in self.df.columns:
            self.df["localization"] = self.df["localization"].fillna("unknown").astype(str).str.strip().str.lower()

        # Impute missing patient age with median
        if "age" in self.df.columns:
            median_age = self.df["age"].median()
            self.df["age"] = self.df["age"].fillna(median_age).astype(float)

    def get_class_distribution(self) -> Dict[str, int]:
        """Compute the sample count per diagnosis category."""
        if self.df is None:
            self.load_metadata()
        return self.df["dx"].value_counts().to_dict()

    def get_patient_lesion_stats(self) -> Dict[str, Union[int, float]]:
        """Compute statistics regarding images per lesion (identifying multi-image lesions)."""
        if self.df is None:
            self.load_metadata()
        images_per_lesion = self.df.groupby("lesion_id")["image_id"].count()
        return {
            "total_images": len(self.df),
            "unique_lesions": len(images_per_lesion),
            "multi_image_lesions": int((images_per_lesion > 1).sum()),
            "max_images_single_lesion": int(images_per_lesion.max()),
            "mean_images_per_lesion": float(images_per_lesion.mean()),
        }
