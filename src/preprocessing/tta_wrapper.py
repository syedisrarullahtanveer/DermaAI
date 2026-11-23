"""
Test-Time Augmentation (TTA) wrapper: averages predictions across horizontal flip, vertical flip, and rotations.
"""
import numpy as np

class TestTimeAugmentor:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Augmentations: original, hflip, vflip, rot180
        views = [
            image,
            np.fliplr(image),
            np.flipud(image),
            np.rot90(image, 2)
        ]
        preds = [self.predict_fn(np.expand_dims(v, 0))[0] for v in views]
        return np.mean(preds, axis=0)
