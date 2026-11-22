"""Clinical Tabular Preprocessor for Patient Demographic Metadata."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ClinicalTabularPreprocessor:
    """Preprocesses clinical tabular features (age, sex, anatomical localization) for multimodal fusion."""

    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ):
        self.numerical_cols = numerical_cols or ["age"]
        self.categorical_cols = categorical_cols or ["sex", "localization"]
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "ClinicalTabularPreprocessor":
        """Fit scaler on numerical demographic features and encoder on categorical features."""
        # Fit numerical
        num_data = df[self.numerical_cols].fillna(df[self.numerical_cols].median())
        self.scaler.fit(num_data)

        # Fit categorical
        cat_data = df[self.categorical_cols].fillna("unknown").astype(str)
        self.encoder.fit(cat_data)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform patient metadata into a unified feature vector."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        num_data = df[self.numerical_cols].fillna(df[self.numerical_cols].median())
        scaled_num = self.scaler.transform(num_data)

        cat_data = df[self.categorical_cols].fillna("unknown").astype(str)
        encoded_cat = self.encoder.transform(cat_data)

        # Concatenate numerical and categorical features
        feature_vector = np.hstack([scaled_num, encoded_cat]).astype(np.float32)
        return feature_vector

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform metadata DataFrame in a single step."""
        return self.fit(df).transform(df)

    @property
    def output_dim(self) -> int:
        """Return the total feature vector dimensionality after encoding."""
        if not self.is_fitted:
            return 0
        return len(self.numerical_cols) + len(self.encoder.get_feature_names_out())
