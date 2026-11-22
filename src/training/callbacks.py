"""Training Callbacks: Checkpointing, Early Stopping, and Learning Rate Schedules."""

from pathlib import Path
from typing import List
import tensorflow as tf
from tensorflow.keras import callbacks


def build_training_callbacks(
    checkpoint_dir: str = "models/saved_models",
    model_name: str = "best_model",
    monitor: str = "val_loss",
    mode: str = "min",
    patience_early_stopping: int = 10,
    patience_reduce_lr: int = 5,
    reduce_lr_factor: float = 0.2,
    min_lr: float = 1e-6,
) -> List[callbacks.Callback]:
    """Create production training callbacks for checkpointing, dynamic LR scheduling, and CSV logging.

    Args:
        checkpoint_dir: Directory to save model weight checkpoints.
        model_name: Filename prefix for model checkpoint.
        monitor: Metric to track.
        mode: 'min' or 'max' optimization direction.
        patience_early_stopping: Epochs to wait before early stopping.
        patience_reduce_lr: Epochs to wait before decaying learning rate.
        reduce_lr_factor: Multiplier for learning rate reduction.
        min_lr: Lower bound for learning rate.

    Returns:
        List[callbacks.Callback]: List of configured Keras callbacks.
    """
    chk_path = Path(checkpoint_dir)
    chk_path.mkdir(parents=True, exist_ok=True)
    best_weights_file = chk_path / f"{model_name}.keras"
    csv_log_file = chk_path / f"{model_name}_training_history.csv"

    callback_list = [
        # Checkpoint: save only the model with best validation metric
        callbacks.ModelCheckpoint(
            filepath=str(best_weights_file),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        # Early Stopping: prevent overfitting
        callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR on Plateau: fine-grained convergence in narrow loss valleys
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=reduce_lr_factor,
            patience=patience_reduce_lr,
            min_lr=min_lr,
            verbose=1,
        ),
        # CSV Logger: persists epoch-by-epoch loss and accuracy metrics
        callbacks.CSVLogger(
            filename=str(csv_log_file),
            separator=",",
            append=False,
        ),
    ]

    return callback_list
