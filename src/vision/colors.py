import cv2
import numpy as np

from ..config import load

# The largest stage appearing in next (orange). Apple and above never come.
NEXT_MAX_TYPE = 4

FRUIT_NAMES = [
    "cherry",
    "strawberry",
    "grape",
    "dekopon",
    "orange",
    "apple",
    "pear",
    "peach",
    "pineapple",
    "melon",
    "watermelon",
]

# Radius ratio per stage (watermelon = 1.0). The Suika Game hitbox radii
# (16.5 - 129.5) divided as is; this ratio is common even when the skin changes.
# It is not geometric: some adjacent stages such as grape/dekopon and apple/pear differ by only 1.13x.
# Those cannot be split by radius, so color decides.
FRUIT_RELATIVE_RADIUS = [
    0.127,   # cherry
    0.185,   # strawberry
    0.236,   # grape
    0.266,   # dekopon
    0.344,   # orange
    0.440,   # apple
    0.498,   # pear
    0.602,   # peach
    0.683,   # pineapple
    0.849,   # melon
    1.000,   # watermelon
]

# watermelon radius / board width. Overridden by config's
# watermelon_radius_ratio to match measurement.
DEFAULT_WATERMELON_RATIO = 0.24

# The board background (beige). V is not narrowed so it can be removed even when darkened by shadow.
BOARD_BG_HSV = ((10, 0, 55), (35, 100, 255))

# The board's frame lines. Highly saturated and indistinguishable from fruit by color,
# so removal is limited to near the border.
BOARD_FRAME_HSV = ((18, 60, 100), (45, 255, 255))

# Every fruit is vivid, and the difference from the background shows in saturation.
DEFAULT_FRUIT_SATURATION_MIN = 95

COLOR_FAMILIES = {
    "red_orange": [0, 1, 3, 4, 5],
    "purple": [2],
    "yellow_green": [6, 8, 9, 10],
    "pink": [7],
}


def saturated_mask(hsv: np.ndarray) -> np.ndarray:
    """Keep only pixels that look like fruit by saturation."""
    saturation_min = load().get("fruit_saturation_min", DEFAULT_FRUIT_SATURATION_MIN)
    return cv2.inRange(hsv, (0, saturation_min, 45), (180, 255, 255))


def color_family(h: float, s: float) -> str:
    if s < 70:
        return "unknown"
    if 125 <= h <= 155:
        return "purple"
    if 148 <= h <= 172:
        return "pink"
    # VRC: every reddish fruit collapses to H<=22. pear and pineapple sit slightly above at
    # H=27-28, which is the boundary between reddish and yellow-green.
    if h <= 24 or h >= 165:
        return "red_orange"
    if h <= 95:
        return "yellow_green"
    return "unknown"
