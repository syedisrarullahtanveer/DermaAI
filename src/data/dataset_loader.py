"""Dataset Loader and Patient-Aware Stratified Data Pipeline for DermaAI."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

try:
    import tensorflow as tf
except ImportError:
    tf = None


class DermaDatasetLoader:
    """Manages image file discovery, patient-aware splitting, and tf.data pipelines."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_dirs: List[Union[str, Path]],
        target_size: Tuple[int, int] = (224, 224),
        class_mapping: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ):
        self.metadata_df = metadata_df.copy()
        self.image_dirs = [Path(d) for d in image_dirs]
        self.target_size = target_size
        self.seed = seed

        # Default 7-class HAM10000 mapping
        self.class_mapping = class_mapping or {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,
            "mel": 4,
            "nv": 5,
            "vasc": 6,
        }
        self.inv_class_mapping = {v: k for k, v in self.class_mapping.items()}

        self._map_image_paths()

    def _map_image_paths(self) -> None:
        """Map each image_id in the metadata to its physical file path across image directories."""
        image_path_lookup = {}
        for img_dir in self.image_dirs:
            if img_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                    for img_file in img_dir.glob(ext):
                        image_id = img_file.stem
                        image_path_lookup[image_id] = str(img_file)

        # Map to DataFrame
        self.metadata_df["file_path"] = self.metadata_df["image_id"].map(image_path_lookup)
        self.metadata_df["label_idx"] = self.metadata_df["dx"].map(self.class_mapping)

    def create_patient_aware_splits(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test partitions grouped by lesion_id.

        This guarantees that no lesion_id appears in more than one partition,
        strictly preventing data leakage and inflated validation accuracy.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

        df = self.metadata_df.dropna(subset=["label_idx"]).copy()
        df["label_idx"] = df["label_idx"].astype(int)

        # Aggregate unique lesions with their primary diagnosis
        lesion_groups = df.groupby("lesion_id").agg({
            "dx": "first",
            "label_idx": "first",
        }).reset_index()

        # First split: Train vs Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        train_lesions, temp_lesions = train_test_split(
            lesion_groups,
            test_size=temp_ratio,
            stratify=lesion_groups["label_idx"],
            random_state=self.seed,
        )

        # Second split: Val vs Test
        val_sub_ratio = val_ratio / temp_ratio
        val_lesions, test_lesions = train_test_split(
            temp_lesions,
            test_size=(1.0 - val_sub_ratio),
            stratify=temp_lesions["label_idx"],
            random_state=self.seed,
        )

        train_df = df[df["lesion_id"].isin(train_lesions["lesion_id"])].copy()
        val_df = df[df["lesion_id"].isin(val_lesions["lesion_id"])].copy()
        test_df = df[df["lesion_id"].isin(test_lesions["lesion_id"])].copy()

        return train_df, val_df, test_df

    def create_tf_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        is_training: bool = False,
        augment_fn=None,
    ):
        """Construct an optimized tf.data.Dataset pipeline for high-throughput model training."""
        if tf is None:
            raise ImportError("TensorFlow is required to build tf.data.Dataset pipelines.")

        # Ensure valid file paths exist
        valid_df = df.dropna(subset=["file_path"]).copy()
        file_paths = valid_df["file_path"].values.astype(str)
        labels = tf.keras.utils.to_categorical(valid_df["label_idx"].values, num_classes=len(self.class_mapping))

        def _load_image(path, label):
            img_raw = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img_raw, channels=3)
            img = tf.image.resize(img, self.target_size)
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        if is_training:
            dataset = dataset.shuffle(buffer_size=min(len(valid_df), 1024), seed=self.seed)

        dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

        if is_training and augment_fn:
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
