"""Custom Learning Rate Schedulers for DermaAI Training.

Implements:
  - CosineAnnealingSchedule : cosine decay between lr_max and eta_min.
  - WarmupCosineDecay       : linear warmup followed by cosine decay.
  - StepDecay               : step-function drop by a fixed factor.

All schedulers subclass `tf.keras.optimizers.schedules.LearningRateSchedule`
so they can be passed directly to any Keras optimizer.
"""

import math
import tensorflow as tf


class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing without warm restarts (SGDR variant, one cycle).

    Args:
        initial_lr: Peak learning rate at step 0.
        t_max:      Number of epochs for one full cosine cycle.
        eta_min:    Minimum learning rate at the trough.
        steps_per_epoch: Needed to convert epoch-level t_max to step-level.
    """

    def __init__(
        self,
        initial_lr: float = 1e-3,
        t_max: int = 10,
        eta_min: float = 1e-6,
        steps_per_epoch: int = 1,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.t_max = t_max
        self.eta_min = eta_min
        self.steps_per_epoch = steps_per_epoch
        # Total steps for one cycle
        self._t_max_steps = t_max * steps_per_epoch

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        t_max = tf.cast(self._t_max_steps, tf.float32)
        cos_inner = math.pi * step / tf.maximum(t_max, 1.0)
        decayed = (
            self.eta_min
            + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + tf.cos(cos_inner))
        )
        return decayed

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "t_max": self.t_max,
            "eta_min": self.eta_min,
            "steps_per_epoch": self.steps_per_epoch,
        }


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay.

    During the warmup phase the LR increases linearly from 0 to `initial_lr`.
    After warmup, it decays following a cosine curve down to `eta_min`.

    Args:
        initial_lr:     Peak learning rate (reached at end of warmup).
        warmup_epochs:  Number of epochs for linear warmup.
        total_epochs:   Total training epochs.
        eta_min:        Floor learning rate after full cosine decay.
        steps_per_epoch: Converts epoch counts to step counts.
    """

    def __init__(
        self,
        initial_lr: float = 1e-3,
        warmup_epochs: int = 1,
        total_epochs: int = 10,
        eta_min: float = 1e-7,
        steps_per_epoch: int = 1,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.steps_per_epoch = steps_per_epoch

        self._warmup_steps = warmup_epochs * steps_per_epoch
        self._total_steps = total_epochs * steps_per_epoch

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self._warmup_steps, tf.float32)
        total_steps = tf.cast(self._total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.initial_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay after warmup
        decay_steps = total_steps - warmup_steps
        decay_step = step - warmup_steps
        cos_inner = math.pi * decay_step / tf.maximum(decay_steps, 1.0)
        cosine_lr = (
            self.eta_min
            + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + tf.cos(cos_inner))
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "eta_min": self.eta_min,
            "steps_per_epoch": self.steps_per_epoch,
        }


class StepDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Step-function learning rate decay.

    Multiplies the learning rate by `drop_factor` every `drop_every_n_epochs`
    epochs.

    Args:
        initial_lr:          Starting learning rate.
        drop_factor:         Multiplicative factor applied at each step boundary.
        drop_every_n_epochs: How often to drop the LR (in epochs).
        steps_per_epoch:     Converts epoch counts to step counts.
    """

    def __init__(
        self,
        initial_lr: float = 1e-3,
        drop_factor: float = 0.5,
        drop_every_n_epochs: int = 2,
        steps_per_epoch: int = 1,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.drop_every_n_epochs = drop_every_n_epochs
        self.steps_per_epoch = steps_per_epoch
        self._steps_per_drop = drop_every_n_epochs * steps_per_epoch

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        steps_per_drop = tf.cast(self._steps_per_drop, tf.float32)
        exponent = tf.floor(step / tf.maximum(steps_per_drop, 1.0))
        lr = self.initial_lr * tf.pow(self.drop_factor, exponent)
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "drop_factor": self.drop_factor,
            "drop_every_n_epochs": self.drop_every_n_epochs,
            "steps_per_epoch": self.steps_per_epoch,
        }


def build_scheduler(
    scheduler_type: str,
    initial_lr: float,
    steps_per_epoch: int,
    **kwargs,
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """Factory that returns the requested scheduler instance.

    Args:
        scheduler_type:  One of ``cosine_annealing``, ``warmup_cosine``, ``step_decay``.
        initial_lr:      Peak learning rate.
        steps_per_epoch: Needed by all schedulers to convert epochs → steps.
        **kwargs:        Additional constructor keyword arguments forwarded to
                         the chosen scheduler class.

    Returns:
        A concrete :class:`LearningRateSchedule` instance.

    Raises:
        ValueError: If ``scheduler_type`` is not recognised.
    """
    registry = {
        "cosine_annealing": CosineAnnealingSchedule,
        "warmup_cosine": WarmupCosineDecay,
        "step_decay": StepDecay,
    }
    if scheduler_type not in registry:
        raise ValueError(
            f"Unknown scheduler type '{scheduler_type}'. "
            f"Choose from: {list(registry.keys())}"
        )
    cls = registry[scheduler_type]
    return cls(initial_lr=initial_lr, steps_per_epoch=steps_per_epoch, **kwargs)
