"""Helpers that convert streamlit-drawable-canvas output into numpy arrays."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pixelation import to_grayscale

CANVAS_SIZE: int = 280


def canvas_to_rgba(image_data: Optional[np.ndarray]) -> np.ndarray:
    """Return the canvas capture as an (H, W, 4) uint8 RGBA array.

    `image_data` is `CanvasResult.image_data`; it is None until the component
    has reported back at least once, in which case a blank black canvas is returned.
    """
    if image_data is None:
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8)
    arr = np.asarray(image_data)
    return np.clip(arr, 0, 255).astype(np.uint8)


def canvas_to_grayscale(image_data: Optional[np.ndarray]) -> np.ndarray:
    """Return the canvas capture as an (H, W) uint8 grayscale array
    (stroke = high intensity, background = 0)."""
    return to_grayscale(canvas_to_rgba(image_data))
