"""End-to-End Model Training Engine for DermaAI."""

from pathlib import Path
from typing import Any, Dict, Optional
import tensorflow as tf
from src.training.callbacks import build_training_callbacks
from src.training.losses import get_loss_function
from src.utils import setup_logger

logger = setup_logger("DermaAI.Trainer")


class DermaModelTrainer:
    """Orchestrates model compilation, callbacks, class-weighted training, and history recording."""

    def __init__(
        self,
        model: tf.keras.Model,
        learning_rate: float = 1e-4,
        optimizer_name: str = "adam",
        loss_name: str = "focal_loss",
        metrics: Optional[list] = None,
    ):
        self.model = model
        self.learning_rate = learning_rate

        # Initialize Optimizer
        if optimizer_name.lower() == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Initialize Loss
        self.loss_fn = get_loss_function(loss_name=loss_name)

        # Standard Clinical Evaluation Metrics
        self.metrics = metrics or [
            "accuracy",
            tf.keras.metrics.CategoricalAccuracy(name="categorical_acc"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]

        self._compile_model()

    def _compile_model(self) -> None:
        """Compile model with designated optimizer, loss function, and evaluation metrics."""
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics,
        )
        logger.info(f"Model '{self.model.name}' successfully compiled with {self.optimizer.name}.")

    def train(
        self,
        train_data: Any,
        val_data: Any,
        epochs: int = 50,
        class_weights: Optional[Dict[int, float]] = None,
        callbacks: Optional[list] = None,
        checkpoint_dir: str = "models/saved_models",
    ) -> tf.keras.callbacks.History:
        """Execute full training pipeline.

        Args:
            train_data: Training tf.data.Dataset or generator.
            val_data: Validation tf.data.Dataset or generator.
            epochs: Maximum training epochs.
            class_weights: Optional dictionary of class weights for loss scaling.
            callbacks: List of Keras callbacks (if None, standard callbacks are auto-generated).
            checkpoint_dir: Directory for saving model weights.

        Returns:
            tf.keras.callbacks.History: Training history object.
        """
        if callbacks is None:
            callbacks = build_training_callbacks(
                checkpoint_dir=checkpoint_dir,
                model_name=self.model.name,
            )

        logger.info(f"Starting training for {epochs} epochs on model: {self.model.name}")
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("Training loop completed successfully.")
        return history
