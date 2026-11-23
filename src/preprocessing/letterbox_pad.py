"""
Letterbox padding to preserve aspect ratio when resizing dermoscopy images.
"""
import cv2
import numpy as np

def letterbox_pad(image: np.ndarray, target_size: tuple = (224, 224), fill_value: int = 0) -> tuple:
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (target_h - nh) // 2
    bottom = target_h - nh - top
    left = (target_w - nw) // 2
    right = target_w - nw - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[fill_value]*image.shape[2])
    return padded, (scale, top, left)
