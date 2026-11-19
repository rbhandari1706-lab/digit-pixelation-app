"""Pure image-processing helpers for the Digit Pixelation Visualizer.

No Streamlit code lives here: every function takes arrays/images in and
returns arrays/images out, so it can be unit-tested with plain numpy data.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID_SIZE: int = 32          # output resolution (32 x 32)
PAD_RATIO: float = 0.2       # margin on each side, as a fraction of the digit's longest side
INK_THRESHOLD: int = 0       # any pixel brighter than this counts as "ink"
MAX_FONT: int = 9            # overlay font size range (px); auto-shrinks to fit a cell
MIN_FONT: int = 5
GRID_LINE_COLOR: Tuple[int, int, int] = (90, 90, 90)


# --------------------------------------------------------------------------- #
# Step 1: grayscale
# --------------------------------------------------------------------------- #
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGBA / RGB / grayscale array to a 2-D uint8 grayscale array.

    RGBA input is composited over black using its alpha channel, so a white
    stroke on either a black *or* a transparent background ends up bright
    while the background ends up 0.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        if arr.shape[2] == 4:
            rgb = rgb * (arr[..., 3:4].astype(np.float32) / 255.0)
        # ITU-R BT.601 luma weights
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Step 2: threshold + bounding box
# --------------------------------------------------------------------------- #
def find_ink_bbox(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (y0, y1, x0, x1) (end-exclusive) of all non-background pixels,
    or None if nothing has been drawn."""
    mask = gray > INK_THRESHOLD
    if not mask.any():
        return None
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


# --------------------------------------------------------------------------- #
# Step 3: center the digit by its ink bounding box
# --------------------------------------------------------------------------- #
def center_in_square(gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop to the ink bbox, then place the crop in the middle of a black
    square canvas with a margin on every side (MNIST-style framing)."""
    y0, y1, x0, x1 = bbox
    crop = gray[y0:y1, x0:x1]
    h, w = crop.shape
    side = max(h, w)
    margin = max(1, int(round(side * PAD_RATIO)))
    size = side + 2 * margin
    square = np.zeros((size, size), dtype=np.uint8)
    top = (size - h) // 2        # equal space above/below (±1px)
    left = (size - w) // 2       # equal space left/right (±1px)
    square[top:top + h, left:left + w] = crop
    return square


# --------------------------------------------------------------------------- #
# Steps 1-5 combined
# --------------------------------------------------------------------------- #
def process_digit(raw_rgba_array: np.ndarray) -> np.ndarray:
    """Turn a raw canvas capture into a centered 32x32 uint8 grayscale array.

    0 = background, 255 = full-intensity stroke, anti-aliased values between.
    """
    gray = to_grayscale(raw_rgba_array)                  # 1. grayscale
    bbox = find_ink_bbox(gray)                           # 2. bounding box of ink
    if bbox is None:                                     #    nothing drawn -> blank grid
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    square = center_in_square(gray, bbox)                # 3. crop + pad to centered square
    small = cv2.resize(                                  # 4. area-based downsample keeps
        square, (GRID_SIZE, GRID_SIZE),                  #    smooth intensity gradients
        interpolation=cv2.INTER_AREA,
    )
    return small.astype(np.uint8)                        # 5. (32, 32) uint8


# --------------------------------------------------------------------------- #
# Pixel-grid rendering with numeric overlay
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a scalable font at `size` px (DejaVu if present, else Pillow's built-in)."""
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)  # Pillow >= 10.1


@lru_cache(maxsize=None)
def _fit_font(cell_px: int) -> ImageFont.FreeTypeFont:
    """Largest font (MAX_FONT..MIN_FONT) whose widest label, '255', fits the cell."""
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    font = _load_font(MIN_FONT)
    for size in range(MAX_FONT, MIN_FONT - 1, -1):
        font = _load_font(size)
        left, _, right, _ = probe.textbbox((0, 0), "255", font=font)
        if right - left <= cell_px - 1:
            break
    return font


def render_pixel_grid_with_values(pixel_array: np.ndarray, upscale_to: int = 320) -> Image.Image:
    """Render a 32x32 array as a crisp block image with grid lines and the
    0-255 value printed in every cell."""
    if pixel_array.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"Expected {(GRID_SIZE, GRID_SIZE)} array, got {pixel_array.shape}")
    pixel_array = pixel_array.astype(np.uint8)

    # Upscale with NEAREST so each source pixel becomes a sharp square block.
    img = Image.fromarray(pixel_array, mode="L").resize((upscale_to, upscale_to), Image.NEAREST)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    cell = upscale_to / GRID_SIZE

    # Thin grid lines between cells.
    for i in range(GRID_SIZE + 1):
        pos = min(int(round(i * cell)), upscale_to - 1)
        draw.line([(pos, 0), (pos, upscale_to - 1)], fill=GRID_LINE_COLOR, width=1)
        draw.line([(0, pos), (upscale_to - 1, pos)], fill=GRID_LINE_COLOR, width=1)

    # Centered intensity text; white on dark cells, black on light cells.
    font = _fit_font(int(cell))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            v = int(pixel_array[r, c])
            color = (255, 255, 255) if v < 128 else (0, 0, 0)
            draw.text(((c + 0.5) * cell, (r + 0.5) * cell), str(v),
                      fill=color, font=font, anchor="mm")
    return img
