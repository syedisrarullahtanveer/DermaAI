"""
Progressive unfreezing callback for staged transfer learning on dermoscopy backbones.
"""
import tensorflow as tf

class ProgressiveFreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, unfreeze_schedule: dict):
        """
        Args:
            unfreeze_schedule: dict mapping epoch -> number of bottom layers to unfreeze.
        """
        super().__init__()
        self.unfreeze_schedule = unfreeze_schedule

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.unfreeze_schedule:
            num_layers = self.unfreeze_schedule[epoch]
            if self.model is not None:
                for layer in self.model.layers[-num_layers:]:
                    layer.trainable = True
