"""Generate LLM-ready reading plates from noisy ink-probability strips.

The OpenRouter client sends one image per request, so we pack several useful
views into a single composite: raw probability, denoised contrast, binary ink,
and component-filtered ink. This gives the vision model both context and a
higher-SNR view without losing the original evidence.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, restoration


def _to_prob(strip_u8: np.ndarray) -> np.ndarray:
    return np.clip(strip_u8.astype(np.float32) / 255.0, 0.0, 1.0)


def _to_u8(prob: np.ndarray) -> np.ndarray:
    return (np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8)


def _component_filter(prob: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    binary = prob > threshold
    binary = morphology.remove_small_objects(binary, min_size=120)
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.binary_dilation(binary, morphology.disk(1))
    return binary.astype(np.float32)


def _local_contrast(prob: np.ndarray) -> np.ndarray:
    den = restoration.denoise_tv_chambolle(prob, weight=0.04, channel_axis=None)
    den = ndi.median_filter(den, size=3)
    return exposure.equalize_adapthist(np.clip(den, 0, 1), clip_limit=0.015).astype(np.float32)


def build_reading_plate(strip_u8: np.ndarray, *, scale: int = 2) -> Tuple[np.ndarray, bytes]:
    """Return ``(plate_u8, png_bytes)`` for a composite reading plate."""
    prob = _to_prob(strip_u8)
    contrast = _local_contrast(prob)
    try:
        sauvola = filters.threshold_sauvola(contrast, window_size=51, k=0.12)
        binary = (contrast > sauvola).astype(np.float32)
    except Exception:
        binary = (contrast > 0.45).astype(np.float32)
    binary = morphology.remove_small_objects(binary > 0, min_size=100).astype(np.float32)
    binary = morphology.binary_closing(binary > 0, morphology.disk(2)).astype(np.float32)
    components = _component_filter(contrast)

    panels = [
        ("RAW INK PROBABILITY", _to_u8(prob)),
        ("DENOISED LOCAL CONTRAST", _to_u8(contrast)),
        ("HIGH-CONTRAST INK MASK", _to_u8(binary)),
        ("LETTER-SCALE COMPONENTS", _to_u8(components)),
    ]

    h, w = panels[0][1].shape
    label_h = 32
    gap = 8
    plate_w = w
    plate_h = len(panels) * (h + label_h) + (len(panels) - 1) * gap
    plate = Image.new("L", (plate_w, plate_h), 0)
    draw = ImageDraw.Draw(plate)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    y = 0
    for title, arr in panels:
        draw.rectangle([0, y, plate_w, y + label_h], fill=245)
        draw.text((10, y + 8), title, fill=0, font=font)
        y += label_h
        plate.paste(Image.fromarray(arr, mode="L"), (0, y))
        y += h + gap

    if scale != 1:
        plate = plate.resize((plate.width * scale, plate.height * scale), Image.Resampling.NEAREST)

    out = io.BytesIO()
    plate.save(out, format="PNG", optimize=True)
    return np.asarray(plate, dtype=np.uint8), out.getvalue()
