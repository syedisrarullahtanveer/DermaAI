"""
Early stopping with automated best-weight restoration for DermaAI training.
"""
import tensorflow as tf

class EarlyStoppingWithRestoration(tf.keras.callbacks.Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 5, mode: str = "min", min_delta: float = 1e-4):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_weights = None
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            return

        is_better = (val < self.best_value - self.min_delta) if self.mode == "min" else (val > self.best_value + self.min_delta)
        if is_better:
            self.best_value = val
            self.wait = 0
            if self.model is not None:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.model is not None:
                    self.model.stop_training = True
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
