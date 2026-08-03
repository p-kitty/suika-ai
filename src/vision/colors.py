import cv2
import numpy as np

# The largest stage of newly appearing fruits (orange). Apple and above never come.
# Applies to both the waiting fruit and the next bubble.
SPAWN_MAX_TYPE = 4

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

# The final stage (watermelon). No merging beyond this.
MAX_FRUIT_TYPE = len(FRUIT_NAMES) - 1

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

# watermelon radius / board width. The value stretched from the old basis 0.24 by the move to the inside-of-the-wall basis
# (0.2874) turned out too large when verified on the tight wedge case of dropping a dekopon/grape between a melon and a pineapple
# (not only the dekopon but also the grape
# stopped reaching the floor). On the real machine the grape goes through and the dekopon gets stuck, so
# it was retaken as 0.280, inside that threshold band (measured 0.2770-0.2810).
WATERMELON_RADIUS_RATIO = 0.280

# The board background (beige). V is not narrowed so it can be removed even when darkened by shadow.
BOARD_BG_HSV = ((10, 0, 55), (35, 100, 255))

# The board's frame lines. Highly saturated and indistinguishable from fruit by color,
# so removal is limited to near the border.
# The board is a gradient turning warm from top to bottom, dropping to H=16 at the bottom.
# Tightening the lower bound drops the band near the floor from the mask and misses the board's bottom edge.
BOARD_FRAME_HSV = ((14, 60, 100), (45, 255, 255))

# Every fruit is vivid, and the difference from the background shows in saturation.
FRUIT_SATURATION_MIN = 95

# Outside the board (the night sky, inside the next bubble) is dark but highly saturated, so saturation alone cannot cut it.
# Meanwhile pale bright things there (clouds, bubbles, stars) drop out on saturation. Requiring both brightness and
# saturation leaves only fruits.
VIVID_SATURATION_MIN = 130
VIVID_VALUE_MIN = 110

COLOR_FAMILIES = {
    "red_orange": [0, 1, 3, 4, 5],
    "purple": [2],
    "yellow_green": [6, 8, 9, 10],
    "pink": [7],
}


def saturated_mask(hsv: np.ndarray) -> np.ndarray:
    """Keep only pixels that look like fruit by saturation."""
    return cv2.inRange(hsv, (0, FRUIT_SATURATION_MIN, 45), (180, 255, 255))


def vivid_mask(hsv: np.ndarray) -> np.ndarray:
    """Keep only bright, vivid pixels. Used to pick up fruit outside the board."""
    return cv2.inRange(hsv, (0, VIVID_SATURATION_MIN, VIVID_VALUE_MIN), (180, 255, 255))


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
