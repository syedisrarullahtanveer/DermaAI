"""
Monte Carlo Dropout uncertainty estimator for clinical predictions.
"""
import numpy as np
import tensorflow as tf

class MCDropoutEstimator:
    """Run multiple stochastic forward passes with dropout active at test time."""
    def __init__(self, model: tf.keras.Model, num_samples: int = 10):
        self.model = model
        self.num_samples = num_samples

    def predict_with_uncertainty(self, x: np.ndarray) -> dict:
        x_tensor = tf.convert_to_tensor(x)
        samples = []
        for _ in range(self.num_samples):
            pred = self.model(x_tensor, training=True)
            samples.append(pred.numpy())
        
        stacked = np.stack(samples, axis=0) # (num_samples, batch, num_classes)
        mean_pred = np.mean(stacked, axis=0)
        epistemic_variance = np.var(stacked, axis=0)
        return {
            "mean": mean_pred,
            "variance": epistemic_variance,
            "std": np.sqrt(epistemic_variance)
        }
