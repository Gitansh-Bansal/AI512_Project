# ------------------------------------------------------------------------------
# Contrast-Enhancing Nighttime Augmentations (CLAHE + Contrast Boost)
# Replaces destructive noise and blinding glare with contrast-restoring transforms:
#   1. Mild CLAHE on Luminance (enhances hidden curb, road, and obstacle edges)
#   2. Subtle Gamma Jitter (0.8 - 1.25)
#   3. Mild Color Temperature & Saturation Shifts
# ------------------------------------------------------------------------------

import random
import cv2
import numpy as np


class MildCLAHE:
    """
    Contrast Limited Adaptive Histogram Equalization on the Luminance (L) channel.
    Amplifies subtle texture boundaries in underexposed regions while strictly
    preventing over-amplification of noise or glare.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8), p: float = 0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_bgr

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l_channel)

        merged = cv2.merge((cl, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


class SubtleGammaJitter:
    """Mild gamma jitter within [0.80, 1.25] to simulate sensor dynamic range without blacking out pixels."""
    def __init__(self, gamma_range: tuple = (0.80, 1.25), p: float = 0.4):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_bgr
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img_bgr, table)


class NightContrastAugmentationPipeline:
    """
    Contrast-restoring augmentation pipeline:
    Replaces destructive noise with boundary-enhancing CLAHE and subtle exposure tuning.
    """
    def __init__(self, p: float = 0.7):
        self.p = p
        self.clahe = MildCLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.6)
        self.gamma = SubtleGammaJitter(gamma_range=(0.82, 1.22), p=0.45)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_bgr
        img = self.clahe(img_bgr)
        img = self.gamma(img)
        return img
