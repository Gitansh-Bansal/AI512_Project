# ------------------------------------------------------------------------------
# Night-Specific Photometric Augmentations for Low-Light Semantic Segmentation
# Implements:
#  1. Random Non-Linear Gamma Correction (simulates under/over-exposure)
#  2. Low-Light Sensor Shot Noise (Poisson-Gaussian noise concentrated in shadows)
#  3. Specular Glare & Headlight Bloom (radial Gaussian intensity bursts)
#  4. NightAugmentationPipeline (pluggable into dataset transform pipelines)
# ------------------------------------------------------------------------------

import random
import numpy as np
import cv2
import torch


class RandomGammaCorrection:
    """
    Applies non-linear gamma adjustment: I_out = I_in ^ gamma.
    gamma > 1: darkens shadows and midtones (underexposure).
    gamma < 1: brightens shadows (overexposure / night recovery).
    """
    def __init__(self, gamma_range=(0.4, 2.2), p=0.6):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_np

        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        # img_np is uint8 [0, 255] or float [0, 1]
        if img_np.dtype == np.uint8:
            inv_gamma = 1.0 / gamma
            lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(img_np, lut)
        else:
            return np.clip(img_np ** (1.0 / gamma), 0.0, 1.0)


class LowLightSensorNoise:
    """
    Injects signal-dependent sensor noise: high variance in dark areas (high ISO noise),
    lower variance in bright illuminated regions.
    """
    def __init__(self, base_noise_std=12.0, dark_noise_multiplier=2.5, p=0.5):
        self.base_noise_std = base_noise_std
        self.dark_noise_multiplier = dark_noise_multiplier
        self.p = p

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_np

        is_uint8 = img_np.dtype == np.uint8
        float_img = img_np.astype(np.float32) if is_uint8 else img_np * 255.0

        # Calculate luminance [0, 255]
        lum = 0.299 * float_img[:, :, 0] + 0.587 * float_img[:, :, 1] + 0.114 * float_img[:, :, 2]
        # Dark areas get more noise (inverse luminance factor)
        dark_factor = 1.0 + self.dark_noise_multiplier * (1.0 - (lum / 255.0))
        noise = np.random.normal(0, self.base_noise_std, float_img.shape).astype(np.float32)
        noise = noise * dark_factor[:, :, None]

        noisy_img = np.clip(float_img + noise, 0, 255)
        return noisy_img.astype(np.uint8) if is_uint8 else (noisy_img / 255.0)


class SpecularGlareBloom:
    """
    Simulates bright oncoming headlights, street lamps, and wet-asphalt reflections
    via radial Gaussian intensity blooms at random positions.
    """
    def __init__(self, num_spots_range=(1, 3), radius_range=(25, 110), intensity_range=(140, 255), p=0.4):
        self.num_spots_range = num_spots_range
        self.radius_range = radius_range
        self.intensity_range = intensity_range
        self.p = p

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_np

        h, w = img_np.shape[:2]
        is_uint8 = img_np.dtype == np.uint8
        result = img_np.astype(np.float32) if is_uint8 else img_np.copy() * 255.0

        num_spots = random.randint(self.num_spots_range[0], self.num_spots_range[1])
        for _ in range(num_spots):
            # Focus glare primarily in the lower-middle half (headlights, asphalt reflections)
            cx = random.randint(int(w * 0.1), int(w * 0.9))
            cy = random.randint(int(h * 0.35), int(h * 0.85))
            radius = random.randint(self.radius_range[0], self.radius_range[1])
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])

            # Coordinate grids around center
            y_indices, x_indices = np.ogrid[:h, :w]
            dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
            mask = np.exp(-dist_sq / (2.0 * (radius / 2.0) ** 2))

            # Glare has warm/white hue
            color_weights = np.array([0.9, 0.95, 1.0])  # slightly blue-white or yellowish
            if random.random() > 0.5:
                color_weights = np.array([1.0, 0.95, 0.8])  # halogen/street yellow

            bloom = (mask[:, :, None] * intensity * color_weights[None, None, :]).astype(np.float32)
            result = np.clip(result + bloom, 0, 255)

        return result.astype(np.uint8) if is_uint8 else (result / 255.0)


class NightAugmentationPipeline:
    """
    Unified photometric augmentation pipeline that combines:
      - Random Gamma Correction
      - Low-light Sensor Noise
      - Specular Glare & Headlight Bloom
    Operates on uint8 BGR/RGB images before standard mean/std normalization.
    """
    def __init__(self, p=0.8):
        self.p = p
        self.gamma = RandomGammaCorrection(gamma_range=(0.4, 2.2), p=0.7)
        self.noise = LowLightSensorNoise(base_noise_std=10.0, dark_noise_multiplier=2.2, p=0.5)
        self.glare = SpecularGlareBloom(num_spots_range=(1, 3), radius_range=(30, 100), intensity_range=(120, 240), p=0.45)

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_np
        img = self.gamma(img_np)
        img = self.noise(img)
        img = self.glare(img)
        return img
