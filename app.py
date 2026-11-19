"""Streamlit UI for the Digit Pixelation Visualizer (layout + state only)."""
from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from canvas_utils import CANVAS_SIZE, canvas_to_rgba
from pixelation import GRID_SIZE, process_digit, render_pixel_grid_with_values

DISPLAY_SIZE = 320     # rendered size of the pixelated grid (px)
STROKE_WIDTH = 20      # brush width (px); spec range 18-22

st.set_page_config(page_title="Digit Pixelation Visualizer", layout="wide")

# 10 canvases of 280px don't fit side by side in most browsers, so give each
# column a minimum width and let the row scroll horizontally.
st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] { overflow-x: auto; flex-wrap: nowrap !important; }
    div[data-testid="stColumn"], div[data-testid="column"] { min-width: 340px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_grid(pixel_bytes: bytes, size: int) -> Image.Image:
    """Render the numeric pixel grid; cached so unchanged digits aren't redrawn
    every time another canvas triggers a rerun."""
    arr = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE)
    return render_pixel_grid_with_values(arr, upscale_to=size)


# Changing the version number changes every canvas key, which resets them all.
if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0

st.title("Digit Pixelation Visualizer")
st.caption(
    "Draw each digit (mouse or pen). Each drawing is cropped, centered and "
    "downsampled to 32×32; the number in every cell is its 0–255 intensity."
)

if st.button("Clear All"):
    st.session_state.canvas_version += 1  # runs before canvases are created below

columns = st.columns(10)
for digit, col in enumerate(columns):
    with col:
        st.markdown(f"<h3 style='text-align:center'>{digit}</h3>", unsafe_allow_html=True)

        # 1. Drawing canvas: white stroke on black; updates on every stroke.
        result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=STROKE_WIDTH,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_{digit}_{st.session_state.canvas_version}",
        )

        # 2. Arrow + caption
        st.markdown(
            "<div style='text-align:center'><div style='font-size:2rem;line-height:1.2'>↓</div>"
            "<div style='font-size:0.8rem;opacity:0.7'>raw → pixelated</div></div>",
            unsafe_allow_html=True,
        )

        # 3 + 4. Process (blank canvas -> all-zero grid) and render with values.
        pixels = process_digit(canvas_to_rgba(result.image_data))
        st.image(cached_grid(pixels.tobytes(), DISPLAY_SIZE), width=DISPLAY_SIZE)
