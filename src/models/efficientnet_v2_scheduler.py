"""
EfficientNetV2 progressive learning scale schedule (progressive image resolution and regularization).
Reference: Tan & Le, "EfficientNetV2: Smaller Models and Faster Training", ICML 2021.
"""
class EfficientNetV2ProgressiveScheduler:
    def __init__(self, min_res: int = 128, max_res: int = 224, min_dropout: float = 0.1, max_dropout: float = 0.3, total_epochs: int = 50):
        self.min_res = min_res
        self.max_res = max_res
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.total_epochs = total_epochs

    def get_stage_config(self, epoch: int) -> dict:
        progress = min(1.0, max(0.0, epoch / max(1, self.total_epochs - 1)))
        current_res = int(self.min_res + progress * (self.max_res - self.min_res))
        # Round to multiple of 16 for conv efficiency
        current_res = (current_res // 16) * 16
        current_dropout = self.min_dropout + progress * (self.max_dropout - self.min_dropout)
        return {
            "image_size": (current_res, current_res),
            "dropout_rate": round(float(current_dropout), 3)
        }
