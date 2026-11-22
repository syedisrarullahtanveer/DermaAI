"""Unit Tests for Image Resizer and Normalizer."""

import numpy as np
from src.preprocessing.image_transforms import ImageResizer, ImageNormalizer


def test_image_resizer_modes():
    """Verify direct, letterbox, and center-crop resizing modes."""
    # Create non-square rectangular test image [80, 160, 3]
    orig_img = np.zeros((80, 160, 3), dtype=np.uint8)
    orig_img[:, :, 0] = 200

    target_size = (100, 100)

    # 1. Letterbox resize
    resizer_lb = ImageResizer(target_size=target_size, mode="letterbox")
    lb_out = resizer_lb.resize(orig_img)
    assert lb_out.shape == (100, 100, 3)
    # Aspect ratio was 2:1, so width 100 -> height 50, leaving top/bottom padding of 25px
    assert np.all(lb_out[0:20, :, :] == 0) # Padding area
    assert np.all(lb_out[30:70, :, 0] == 200) # Content area

    # 2. Direct resize
    resizer_direct = ImageResizer(target_size=target_size, mode="direct")
    direct_out = resizer_direct.resize(orig_img)
    assert direct_out.shape == (100, 100, 3)

    # 3. Center crop resize
    resizer_crop = ImageResizer(target_size=target_size, mode="center_crop")
    crop_out = resizer_crop.resize(orig_img)
    assert crop_out.shape == (100, 100, 3)


def test_image_normalizer_and_inversion():
    """Verify min_max, zero_center, and imagenet normalizations and their inverses."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128

    # Min-max
    norm_minmax = ImageNormalizer(method="min_max")
    out_mm = norm_minmax.normalize(img)
    assert out_mm.min() >= 0.0 and out_mm.max() <= 1.0
    denorm_mm = norm_minmax.denormalize(out_mm)
    assert np.allclose(denorm_mm, img, atol=1)

    # Zero-center
    norm_zc = ImageNormalizer(method="zero_center")
    out_zc = norm_zc.normalize(img)
    assert out_zc.min() >= -1.0 and out_zc.max() <= 1.0
    denorm_zc = norm_zc.denormalize(out_zc)
    assert np.allclose(denorm_zc, img, atol=1)

    # ImageNet
    norm_in = ImageNormalizer(method="imagenet")
    out_in = norm_in.normalize(img)
    assert out_in.shape == (50, 50, 3)
    denorm_in = norm_in.denormalize(out_in)
    assert np.allclose(denorm_in, img, atol=2)
