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

# Radius ratio per stage (watermelon = 1.0). The Suika Game hitbox
# grows about 1.2x per stage, and this ratio is common even when the skin changes.
FRUIT_RELATIVE_RADIUS = [
    0.084,   # cherry
    0.130,   # strawberry
    0.175,   # grape
    0.234,   # dekopon
    0.299,   # orange
    0.383,   # apple
    0.481,   # pear
    0.591,   # peach
    0.721,   # pineapple
    0.838,   # melon
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
    if 30 <= h <= 95:
        return "yellow_green"
    # VRC: every reddish fruit gathers at H≈15-25
    if h <= 28 or h >= 165:
        return "red_orange"
    if 15 <= h <= 35:
        return "yellow_green"
    return "unknown"
