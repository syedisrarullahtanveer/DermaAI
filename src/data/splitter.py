"""Patient-Aware Stratified Dataset Partitioning and Leakage Verification Module."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from src.utils import setup_logger

logger = setup_logger("DermaAI.DataSplitter")


class PatientAwareDataSplitter:
    """Partitions dermoscopy datasets grouped strictly by patient/lesion identifier to prevent data leakage."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        lesion_id_col: str = "lesion_id",
        label_col: str = "dx",
        seed: int = 42,
    ):
        self.df = metadata_df.copy()
        self.lesion_id_col = lesion_id_col
        self.label_col = label_col
        self.seed = seed

    def split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute patient-grouped stratified split.

        Args:
            train_ratio: Fraction of dataset allocated to training.
            val_ratio: Fraction allocated to validation.
            test_ratio: Fraction allocated to testing.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

        df_valid = self.df.dropna(subset=[self.lesion_id_col, self.label_col]).copy()

        # Step 1: Aggregate by lesion_id so each lesion appears exactly once in the split pool
        lesion_summary = df_valid.groupby(self.lesion_id_col).agg({
            self.label_col: "first",
        }).reset_index()

        # Step 2: Split unique lesions into Train vs Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        train_lesions, temp_lesions = train_test_split(
            lesion_summary,
            test_size=temp_ratio,
            stratify=lesion_summary[self.label_col],
            random_state=self.seed,
        )

        # Step 3: Split Temp into Validation vs Test
        val_sub_ratio = val_ratio / temp_ratio
        val_lesions, test_lesions = train_test_split(
            temp_lesions,
            test_size=(1.0 - val_sub_ratio),
            stratify=temp_lesions[self.label_col],
            random_state=self.seed,
        )

        # Step 4: Map partitioned lesions back to all corresponding image rows
        train_df = df_valid[df_valid[self.lesion_id_col].isin(train_lesions[self.lesion_id_col])].copy()
        val_df = df_valid[df_valid[self.lesion_id_col].isin(val_lesions[self.lesion_id_col])].copy()
        test_df = df_valid[df_valid[self.lesion_id_col].isin(test_lesions[self.lesion_id_col])].copy()

        # Assign explicit partition label
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        # Step 5: Verify zero patient leakage
        self.verify_no_data_leakage(train_df, val_df, test_df)

        logger.info(
            f"Dataset split complete: Train={len(train_df)} ({len(train_lesions)} lesions), "
            f"Val={len(val_df)} ({len(val_lesions)} lesions), "
            f"Test={len(test_df)} ({len(test_lesions)} lesions)."
        )

        return train_df, val_df, test_df

    def verify_no_data_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> bool:
        """Mathematically assert that the intersection of lesion IDs between all partitions is empty."""
        train_lesions = set(train_df[self.lesion_id_col].dropna().unique())
        val_lesions = set(val_df[self.lesion_id_col].dropna().unique())
        test_lesions = set(test_df[self.lesion_id_col].dropna().unique())

        leak_train_val = train_lesions.intersection(val_lesions)
        leak_train_test = train_lesions.intersection(test_lesions)
        leak_val_test = val_lesions.intersection(test_lesions)

        if leak_train_val or leak_train_test or leak_val_test:
            error_msg = (
                f"PATIENT DATA LEAKAGE DETECTED! "
                f"Train/Val overlap={len(leak_train_val)}, "
                f"Train/Test overlap={len(leak_train_test)}, "
                f"Val/Test overlap={len(leak_val_test)}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("[x] Leakage Verification Passed: Zero lesion overlap detected between splits.")
        return True

    def export_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Union[str, Path] = "dataset/splits",
    ) -> Dict[str, str]:
        """Save split dataframes to CSV manifests on disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        train_file = out_path / "train.csv"
        val_file = out_path / "val.csv"
        test_file = out_path / "test.csv"

        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)

        logger.info(f"Splits saved to {out_path}: train.csv, val.csv, test.csv")
        return {
            "train": str(train_file),
            "val": str(val_file),
            "test": str(test_file),
        }
