"""Hand-crafted features for the classic baseline: colour histogram + HOG (edge-direction histogram).

If this already gets most of the way, the CNN has to justify itself.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog

from . import preprocess

SMALL = 64         # features are computed on a 64 px centre crop: HOG on 128 px is slow and no better


def _small(img: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(preprocess.center_crop(img)).resize((SMALL, SMALL), Image.BILINEAR))


def color_hist(img: np.ndarray, bins: int = 16) -> np.ndarray:
    hsv = rgb2hsv(img)
    return np.concatenate([np.histogram(hsv[..., c], bins=bins, range=(0, 1), density=True)[0] for c in range(3)])


def hog_vec(img: np.ndarray) -> np.ndarray:
    return hog(rgb2gray(img), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)


def extract(imgs: np.ndarray) -> np.ndarray:
    """(N, SIZE, SIZE, 3) uint8 -> (N, 48 + 1764) float32."""
    rows = []
    for im in imgs:
        s = _small(im)
        rows.append(np.concatenate([color_hist(s), hog_vec(s)]))
    return np.asarray(rows, dtype=np.float32)
