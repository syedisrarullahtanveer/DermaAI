"""
Gradient accumulation helper for simulating large batch sizes under memory constraints.
"""
import tensorflow as tf

class GradientAccumulator:
    def __init__(self, steps: int = 4):
        self.steps = steps
        self.accumulated_grads = None
        self.current_step = 0

    def accumulate(self, grads):
        if self.accumulated_grads is None:
            self.accumulated_grads = [tf.Variable(tf.zeros_like(g), trainable=False) for g in grads]
        for a, g in zip(self.accumulated_grads, grads):
            a.assign_add(g)
        self.current_step += 1

    def should_apply(self) -> bool:
        return self.current_step >= self.steps

    def get_and_reset(self):
        scaled_grads = [a / float(self.steps) for a in self.accumulated_grads]
        for a in self.accumulated_grads:
            a.assign(tf.zeros_like(a))
        self.current_step = 0
        return scaled_grads
