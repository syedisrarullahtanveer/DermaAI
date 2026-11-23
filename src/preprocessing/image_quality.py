import numpy as np

def is_blurry(image: np.ndarray, threshold: float = 10.0) -> bool:
    """Return ``True`` if the image appears blurry.

    A very simple heuristic is used: the standard deviation of pixel
    intensities is computed across the whole image.  Low variance
    typically indicates a uniform (and thus likely blurry) image.

    Parameters
    ----------
    image: np.ndarray
        Input image as a ``(H, W, C)`` array or a 2‑D grayscale array.
    threshold: float, optional
        Standard‑deviation threshold below which the image is considered
        blurry.  The default works well for 8‑bit images.
    """
    if image.ndim == 3:
        # Convert to grayscale by averaging channels
        image = image.mean(axis=2)
    std = np.std(image)
    return std < threshold
