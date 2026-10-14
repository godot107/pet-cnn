"""Image -> model input. numpy + Pillow only, so the Lambda handler ships this file unchanged.

Every path (cache build, training, evaluation, ONNX check, Lambda) goes through these constants.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 144                      # cached square: shorter side resized to SIZE, then centre-cropped
CROP = 128                      # model input; training takes random CROP crops of the SIZE square
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet stats (resnet18 expects them)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def square(img: Image.Image, size: int = SIZE) -> np.ndarray:
    """RGB uint8 (size, size, 3): shorter side to `size`, centre crop."""
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR, reducing_gap=3.0)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.uint8)


def center_crop(x: np.ndarray, crop: int = CROP) -> np.ndarray:
    """(..., H, W, C) -> (..., crop, crop, C)."""
    h, w = x.shape[-3:-1]
    t, l = (h - crop) // 2, (w - crop) // 2
    return x[..., t:t + crop, l:l + crop, :]


def normalize(x: np.ndarray) -> np.ndarray:
    """uint8 NHWC -> float32 NCHW, ImageNet-normalised."""
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def to_input(img: Image.Image) -> np.ndarray:
    """One PIL image -> (1, 3, CROP, CROP) float32, exactly as evaluation sees it."""
    return normalize(center_crop(square(img))[None])
