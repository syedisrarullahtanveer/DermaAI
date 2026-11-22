"""Experiment Runner for DermaAI Multi-Model Training.

Orchestrates the full training loop for a single model experiment:
  1. Loads configuration from configs/training_experiments.yaml.
  2. Builds and compiles the requested model architecture.
  3. Sets up callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
     TensorBoard, CSVLogger).
  4. Trains on the DermaAI synthetic / real dataset.
  5. Captures the actual Keras History object.
  6. Saves JSON results and training curve PNGs.
  7. Returns a typed result dict for downstream comparison.

Usage:
    runner = ExperimentRunner("configs/training_experiments.yaml")
    result = runner.run("baseline_cnn")
    result = runner.run("resnet50")
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
import yaml

# ---------------------------------------------------------------------------
# Ensure project root on path so src.* imports work from any cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.lr_scheduler import build_scheduler
from src.training.history_visualizer import plot_training_curves

# ---------------------------------------------------------------------------
# Lazy import of model builders (avoids heavy TF imports at module load time)
# ---------------------------------------------------------------------------


def _build_model(model_name: str, num_classes: int, image_size: tuple, exp_cfg: dict):
    """Dispatch to the correct model factory based on ``model_name``."""
    if model_name == "baseline_cnn":
        from src.models.baseline_cnn import build_baseline_cnn
        return build_baseline_cnn(
            input_shape=(*image_size, 3),
            num_classes=num_classes,
            dropout_rate=exp_cfg.get("dropout", 0.4),
            l2_reg=exp_cfg.get("l2_reg", 1e-4),
        )
    elif model_name == "resnet50":
        from src.models.resnet_model import build_resnet50
        return build_resnet50(
            input_shape=(*image_size, 3),
            num_classes=num_classes,
            dropout_rate=exp_cfg.get("dropout", 0.4),
            freeze_backbone=exp_cfg.get("freeze_backbone", True),
        )
    elif model_name == "efficientnet_b0":
        from src.models.efficientnet_model import build_efficientnet_b0
        return build_efficientnet_b0(
            input_shape=(*image_size, 3),
            num_classes=num_classes,
            dropout_rate=exp_cfg.get("dropout", 0.3),
            freeze_backbone=exp_cfg.get("freeze_backbone", True),
        )
    elif model_name == "mobilenet_v2":
        from src.models.mobilenet_model import build_mobilenet_v2
        return build_mobilenet_v2(
            input_shape=(*image_size, 3),
            num_classes=num_classes,
            dropout_rate=exp_cfg.get("dropout", 0.3),
            freeze_backbone=exp_cfg.get("freeze_backbone", True),
        )
    elif model_name.startswith("hybrid") or "hybrid" in model_name:
        from src.models.hybrid_model import build_hybrid_model
        return build_hybrid_model(
            input_shape=(*image_size, 3),
            num_classes=num_classes,
            backbone=exp_cfg.get("backbone", "efficientnet_b0"),
            attention_type=exp_cfg.get("attention_type", "cbam"),
            use_metadata=exp_cfg.get("use_metadata", False),
            metadata_dim=exp_cfg.get("metadata_dim", 10),
            fusion_type=exp_cfg.get("fusion_type", "gated"),
            dropout_rate=exp_cfg.get("dropout", 0.35),
            freeze_backbone=exp_cfg.get("freeze_backbone", True),
        )
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")


def _build_dataset(
    image_size: tuple,
    batch_size: int,
    seed: int,
    *,
    split_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    num_classes: int = 7,
    use_metadata: bool = False,
):
    """Create TF datasets for train / validation.

    If real split CSVs are found (dataset/splits/), they are used.
    Otherwise falls back to a tf.data pipeline over the synthetic sample images.
    """
    # Try loading from dataset splits
    splits_dir = Path(split_dir) if split_dir else _PROJECT_ROOT / "dataset" / "splits"
    img_dir = Path(image_dir) if image_dir else _PROJECT_ROOT / "dataset" / "raw" / "sample_images"

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if train_csv.exists() and val_csv.exists() and img_dir.exists():
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        # Support both 'label' and 'dx' (HAM10000-style) column names
        label_col = "label" if "label" in train_df.columns else "dx"

        # Map string label → integer index
        all_labels = sorted(train_df[label_col].unique().tolist())
        label2idx = {l: i for i, l in enumerate(all_labels)}

        # Preprocess metadata if requested
        meta_dim = 0
        train_meta_vecs = None
        val_meta_vecs = None
        if use_metadata:
            from src.preprocessing.tabular_preprocessor import ClinicalTabularPreprocessor
            preprocessor = ClinicalTabularPreprocessor()
            train_meta_vecs = preprocessor.fit_transform(train_df)
            val_meta_vecs = preprocessor.transform(val_df)
            meta_dim = preprocessor.output_dim
            train_df = train_df.copy()
            train_df["_meta_idx"] = np.arange(len(train_df))
            val_df = val_df.copy()
            val_df["_meta_idx"] = np.arange(len(val_df))

        def _load_df(df, meta_vecs=None, augment=False):
            paths = []
            labels = []
            metas = []
            for _, row in df.iterrows():
                img_path = img_dir / f"{row['image_id']}.jpg"
                if img_path.exists():
                    paths.append(str(img_path))
                    labels.append(label2idx.get(row[label_col], 0))
                    if use_metadata and meta_vecs is not None:
                        metas.append(meta_vecs[row["_meta_idx"]])

            if not paths:
                return None, 0

            path_ds = tf.data.Dataset.from_tensor_slices(paths)
            label_ds = tf.data.Dataset.from_tensor_slices(
                tf.keras.utils.to_categorical(labels, num_classes=num_classes)
            )

            def _decode(p):
                raw = tf.io.read_file(p)
                img = tf.image.decode_jpeg(raw, channels=3)
                img = tf.image.resize(img, image_size)
                img = tf.cast(img, tf.float32) / 255.0
                return img

            img_ds = path_ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)

            if use_metadata and metas:
                meta_ds = tf.data.Dataset.from_tensor_slices(
                    np.array(metas, dtype=np.float32)
                )
                if augment:
                    # Apply image augmentation while keeping metadata unchanged
                    def _aug_multimodal(x, m):
                        x = tf.image.random_flip_left_right(x)
                        x = tf.image.random_flip_up_down(x)
                        return (x, m)
                    input_ds = tf.data.Dataset.zip((img_ds, meta_ds)).map(
                        _aug_multimodal, num_parallel_calls=tf.data.AUTOTUNE
                    )
                else:
                    input_ds = tf.data.Dataset.zip((img_ds, meta_ds))
                ds = tf.data.Dataset.zip((input_ds, label_ds))
            else:
                if augment:
                    img_ds = img_ds.map(
                        lambda x: tf.image.random_flip_left_right(
                            tf.image.random_flip_up_down(x)
                        ),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
                ds = tf.data.Dataset.zip((img_ds, label_ds))

            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), len(paths)

        train_ds, n_train = _load_df(train_df, meta_vecs=train_meta_vecs, augment=True)
        val_ds, n_val = _load_df(val_df, meta_vecs=val_meta_vecs, augment=False)
        return train_ds, val_ds, n_train, n_val, meta_dim

    # ── Fallback: build a minimal dataset from sample images on disk ──────
    fb_train, fb_val, fb_nt, fb_nv = _build_fallback_dataset(img_dir, image_size, batch_size, num_classes)
    return fb_train, fb_val, fb_nt, fb_nv, 0


def _build_fallback_dataset(img_dir: Path, image_size, batch_size, num_classes):
    """Minimal tf.data pipeline from sample images on disk.

    Expects images to be named ``<class>_NNNNN.jpg`` matching the
    class directory structure created by create_sample_dataset.py.
    """
    class_names = sorted(
        [d.name for d in img_dir.iterdir() if d.is_dir()]
    ) if img_dir.is_dir() else []

    if not class_names:
        # As a last resort, generate random data to verify the pipeline
        print("[ExperimentRunner] WARNING: No dataset found. Using random noise data.")
        n_train, n_val = 56, 14
        h, w = image_size
        x_train = np.random.rand(n_train, h, w, 3).astype(np.float32)
        y_train = tf.keras.utils.to_categorical(
            np.random.randint(0, num_classes, n_train), num_classes
        )
        x_val = np.random.rand(n_val, h, w, 3).astype(np.float32)
        y_val = tf.keras.utils.to_categorical(
            np.random.randint(0, num_classes, n_val), num_classes
        )
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((x_val, y_val))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return train_ds, val_ds, n_train, n_val

    label2idx = {c: i for i, c in enumerate(class_names)}
    all_paths, all_labels = [], []
    for cls_dir in img_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        idx = label2idx.get(cls_dir.name, 0)
        for img_path in cls_dir.glob("*.jpg"):
            all_paths.append(str(img_path))
            all_labels.append(idx)

    # Shuffle with fixed seed
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_paths))
    all_paths = [all_paths[i] for i in perm]
    all_labels = [all_labels[i] for i in perm]

    split = max(1, int(0.8 * len(all_paths)))
    train_p, val_p = all_paths[:split], all_paths[split:]
    train_l, val_l = all_labels[:split], all_labels[split:]

    def _make_ds(paths, labels, augment=False):
        path_ds = tf.data.Dataset.from_tensor_slices(paths)
        label_ds = tf.data.Dataset.from_tensor_slices(
            tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        )

        def _decode(p):
            raw = tf.io.read_file(p)
            img = tf.image.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, image_size)
            img = tf.cast(img, tf.float32) / 255.0
            return img

        img_ds = path_ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        ds = tf.data.Dataset.zip((img_ds, label_ds))
        if augment:
            ds = ds.map(
                lambda x, y: (tf.image.random_flip_left_right(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return (
        _make_ds(train_p, train_l, augment=True),
        _make_ds(val_p, val_l, augment=False),
        len(train_p),
        len(val_p),
    )


class ExperimentRunner:
    """Orchestrates training experiments for multiple DermaAI architectures.

    Args:
        config_path: Path to ``configs/training_experiments.yaml``.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(_PROJECT_ROOT / "configs" / "training_experiments.yaml")
        with open(config_path, "r") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        self.common = self.cfg["common"]
        self.experiments = self.cfg["experiments"]
        self.lr_cfgs = self.cfg.get("learning_rate_schedulers", {})

        # Directories
        self.results_dir = _PROJECT_ROOT / self.common["results_dir"]
        self.plots_dir = _PROJECT_ROOT / self.common["plots_dir"]
        self.ckpt_dir = _PROJECT_ROOT / self.common["checkpoint_dir"]
        for d in (self.results_dir, self.plots_dir, self.ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(
        self,
        model_name: str,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train a single model and return the result dict.

        Args:
            model_name:  Key in ``experiments`` section of the YAML config.
            epochs:      Override the config epoch count (useful for quick tests).
            batch_size:  Override batch size.

        Returns:
            Dict with keys: model_name, epochs, train_loss, train_accuracy,
            val_loss, val_accuracy, duration_s, history, plot_path.
        """
        if model_name not in self.experiments:
            raise ValueError(
                f"Unknown experiment '{model_name}'. "
                f"Available: {list(self.experiments.keys())}"
            )

        exp_cfg = self.experiments[model_name]
        num_classes = self.common["num_classes"]
        image_size = tuple(self.common["image_size"])
        n_epochs = epochs or self.common["epochs"]
        bs = batch_size or self.common["batch_size"]
        seed = self.common["seed"]

        tf.random.set_seed(seed)
        np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {model_name}")
        print(f"  Epochs: {n_epochs}  |  Batch size: {bs}")
        print(f"{'='*60}")

        # 1. Dataset
        print("[1/5] Loading dataset …")
        train_ds, val_ds, n_train, n_val = _build_dataset(
            image_size=image_size,
            batch_size=bs,
            seed=seed,
            num_classes=num_classes,
        )
        steps_per_epoch = max(1, n_train // bs)
        print(f"      Train: {n_train} samples  |  Val: {n_val} samples")

        # 2. Model
        print("[2/5] Building model …")
        model = _build_model(model_name, num_classes, image_size, exp_cfg)

        # 3. Scheduler + Optimizer
        print("[3/5] Configuring optimizer and LR scheduler …")
        scheduler_key = exp_cfg.get("lr_scheduler", "cosine_annealing")
        lr_cfg = self.lr_cfgs.get(scheduler_key, {})
        initial_lr = exp_cfg.get("lr", exp_cfg.get("lr_stage1", 1e-3))
        scheduler = build_scheduler(
            scheduler_type=lr_cfg.get("type", scheduler_key),
            initial_lr=initial_lr,
            steps_per_epoch=steps_per_epoch,
            **{
                k: v
                for k, v in lr_cfg.items()
                if k not in ("type", "initial_lr")
            },
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

        # 4. Loss
        loss_name = self.common.get("loss", "categorical_crossentropy")
        from src.training.losses import get_loss_function
        loss_fn = get_loss_function(
            loss_name=loss_name,
            gamma=self.common.get("focal_gamma", 2.0),
            alpha=self.common.get("focal_alpha", 0.25),
        )

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"],
        )

        # 5. Callbacks
        print("[4/5] Setting up callbacks …")
        ckpt_path = str(
            self.ckpt_dir / f"{model_name}_best.keras"
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=exp_cfg.get("early_stopping_patience", 10),
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            tf.keras.callbacks.CSVLogger(
                str(self.results_dir / f"{model_name}_training_log.csv")
            ),
        ]
        # Only add ReduceLROnPlateau if learning rate is a scalar, not a step-based schedule
        if not isinstance(scheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=exp_cfg.get("reduce_lr_patience", 3),
                    min_lr=1e-7,
                    verbose=1,
                )
            )

        # 6. Train
        print("[5/5] Training …")
        t0 = time.time()
        history_obj = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=n_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        duration = time.time() - t0
        hist = history_obj.history  # dict of lists

        # 7. Persist results
        result = {
            "model_name": model_name,
            "description": exp_cfg.get("description", ""),
            "epochs_run": len(hist.get("loss", [])),
            "epochs_requested": n_epochs,
            "train_samples": n_train,
            "val_samples": n_val,
            "batch_size": bs,
            "train_loss": hist.get("loss", []),
            "train_accuracy": hist.get("accuracy", hist.get("acc", [])),
            "val_loss": hist.get("val_loss", []),
            "val_accuracy": hist.get("val_accuracy", hist.get("val_acc", [])),
            "best_val_loss": min(hist.get("val_loss", [float("inf")])),
            "best_val_accuracy": max(hist.get("val_accuracy", hist.get("val_acc", [0]))),
            "duration_seconds": round(duration, 2),
            "checkpoint_path": ckpt_path,
        }

        json_path = self.results_dir / f"{model_name}_results.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved → {json_path}")

        # 8. Training curves
        plot_path = plot_training_curves(
            history=hist,
            model_name=model_name,
            output_dir=str(self.plots_dir),
        )
        result["plot_path"] = plot_path
        print(f"  Plot saved    → {plot_path}")

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    def run_all(
        self,
        model_names: Optional[List[str]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """Run experiments for all (or specified) models sequentially.

        Args:
            model_names: Subset of experiment keys to run.  Defaults to all.
            epochs:      Epoch override forwarded to each :meth:`run` call.
            batch_size:  Batch-size override.

        Returns:
            Mapping ``{model_name: result_dict}``.
        """
        targets = model_names or list(self.experiments.keys())
        all_results = {}
        for name in targets:
            try:
                all_results[name] = self.run(name, epochs=epochs, batch_size=batch_size)
            except Exception as exc:  # noqa: BLE001
                print(f"\n[ExperimentRunner] ERROR running '{name}': {exc}")
                all_results[name] = {"error": str(exc)}
        return all_results

    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(result: Dict[str, Any]) -> None:
        name = result["model_name"]
        epochs = result.get("epochs_run", "?")
        best_val_acc = result.get("best_val_accuracy", 0.0)
        best_val_loss = result.get("best_val_loss", float("inf"))
        dur = result.get("duration_seconds", 0)
        print(f"\n  ┌─ {name} summary ─────────────────────────────")
        print(f"  │  Epochs run       : {epochs}")
        print(f"  │  Best val loss    : {best_val_loss:.4f}")
        print(f"  │  Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
        print(f"  │  Duration         : {dur:.1f}s")
        print(f"  └─────────────────────────────────────────────")
