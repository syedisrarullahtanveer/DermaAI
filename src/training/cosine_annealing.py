"""Cosine Annealing with Warm Restarts (SGDR) Learning Rate Schedule.

Reference:
    Loshchilov, I., & Hutter, F. (2017).
    SGDR: Stochastic Gradient Descent with Warm Restarts. In ICLR 2017.
"""

import math
from typing import Dict, Any
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="DermaAI")
class CosineAnnealingWarmRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine Annealing schedule that periodically resets learning rate with optional cycle multiplier."""

    def __init__(
        self,
        initial_learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-6,
        first_cycle_steps: int = 1000,
        cycle_mult: float = 1.5,
        name: str = "CosineAnnealingWarmRestarts",
    ) -> None:
        """Initialize scheduler.

        Args:
            initial_learning_rate: Peak learning rate at the start of each cycle.
            min_learning_rate: Minimum learning rate floor at the end of each cycle.
            first_cycle_steps: Number of steps in the initial cycle T_0.
            cycle_mult: Factor T_mult by which the cycle length scales on each restart.
            name: String name of schedule.
        """
        super().__init__()
        self.initial_learning_rate = float(initial_learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        self.first_cycle_steps = int(first_cycle_steps)
        self.cycle_mult = float(cycle_mult)
        self.name = name

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """Compute the learning rate at the given global training step."""
        step = tf.cast(step, tf.float32)
        t_0 = tf.cast(self.first_cycle_steps, tf.float32)
        mult = tf.cast(self.cycle_mult, tf.float32)

        if self.cycle_mult == 1.0:
            cycle = tf.math.floor(step / t_0)
            t_cur = step - cycle * t_0
            t_i = t_0
        else:
            # Solve for cycle index i: t_0 * (mult^i - 1) / (mult - 1) <= step
            cycle = tf.math.floor(
                tf.math.log(step * (mult - 1.0) / t_0 + 1.0) / tf.math.log(mult)
            )
            # Cycle start step
            cycle_start = t_0 * (tf.math.pow(mult, cycle) - 1.0) / (mult - 1.0)
            t_i = t_0 * tf.math.pow(mult, cycle)
            t_cur = step - cycle_start

        # Fraction completed in current cycle: [0, 1]
        fraction = tf.clip_by_value(t_cur / t_i, 0.0, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.math.cos(math.pi * fraction))
        lr = self.min_learning_rate + (self.initial_learning_rate - self.min_learning_rate) * cosine_decay
        return lr

    def get_config(self) -> Dict[str, Any]:
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "first_cycle_steps": self.first_cycle_steps,
            "cycle_mult": self.cycle_mult,
            "name": self.name,
        }
