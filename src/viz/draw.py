import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

Color = tuple[int, int, int]
Point = tuple[int, int]

# BGR per stage. Colors sampled near the center of the boards in screenshots.
# Shared by the mini boards of view_sim / preview.
FRUIT_BGR: list[Color] = [
    (2, 5, 199),       # cherry — deep red
    (59, 90, 208),     # strawberry — strongly reddish orange-red
    (212, 88, 134),    # grape — purple
    (5, 155, 211),     # dekopon — bright orange
    (19, 113, 216),    # orange — slightly reddish orange (persimmon)
    (17, 15, 204),     # apple — deep red
    (104, 202, 211),   # pear — pale yellow-green
    (145, 153, 213),   # peach — pink
    (4, 196, 206),     # pineapple — yellow
    (12, 185, 130),    # melon — yellow-green
    (6, 127, 14),      # watermelon — deep green
]


def put_text(
    image: np.ndarray,
    text: str,
    origin: Point,
    color: Color,
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def mode_badge(image: np.ndarray, auto: bool, fast: bool = False) -> None:
    """Always show AUTO / LIVE (+FAST) at the top right of the screen. Does not overlap board/next at the top left."""
    label = "AUTO" if auto else "LIVE"
    if fast:
        label += "+FAST"
    fg: Color = (40, 255, 120) if auto else (170, 170, 170)
    bg: Color = (20, 70, 30) if auto else (36, 36, 36)
    border: Color = (60, 255, 160) if auto else (90, 90, 90)
    if fast:
        fg, bg, border = (60, 220, 255), (20, 50, 70), (80, 200, 255)
    scale = 1.15
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(label, FONT, scale, thickness)
    pad_x, pad_y = 12, 8
    x = image.shape[1] - tw - pad_x - 8
    y = 14 + th
    top_left = (x - pad_x, y - th - pad_y)
    bottom_right = (x + tw + pad_x, y + baseline + pad_y // 2)
    cv2.rectangle(image, top_left, bottom_right, bg, -1)
    cv2.rectangle(image, top_left, bottom_right, border, 2)
    put_text(image, label, (x, y), fg, scale=scale, thickness=thickness)
