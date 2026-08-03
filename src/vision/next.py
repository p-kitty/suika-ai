import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..draw import Color, put_text
from .blobs import circle_peaks, solid_mask
from .classify import ClassifyResult, classify, fruit_radius_ratios, sample_hsv
from .colors import SPAWN_MAX_TYPE, vivid_mask
from .normalized import (
    NORMALIZED_WIDTH,
    inverse_warp_matrix,
    screen_circle,
    warp_window,
)

# The bubble floats around the top right of the board. As its center viewed as a shadow cast onto the board's plane,
# Measured, all 11 images fall within (538-557, -10 to 10).
BUBBLE_X = 550
BUBBLE_Y = 5

# Radius of the window taken around the center. Wide enough to exclude the Merge Order ring below and the tree on the right.
WINDOW_HALF = 100
# A blob this far from the bubble's center is not the next fruit.
CENTER_TOLERANCE = 45


@dataclass
class NextResult:
    """The contents of the next bubble. The fruit that comes after the waiting one."""

    fruit: ClassifyResult | None
    # Coordinates on the normalized board. Outside the board to the right, so x is larger than the width.
    x: float | None = None
    y: float | None = None
    radius: float | None = None
    radius_ratio: float | None = None


def detect(frame: np.ndarray, corners: np.ndarray) -> NextResult:
    window = _warp_window(frame, corners)
    mask = _window_mask(window)

    blob = _find_blob(mask)
    if blob is None:
        return NextResult(fruit=None)

    x, y, radius = blob
    # Fruit inside the bubble appears at almost the same size as when placed on the board
    # (median 1.00x over 11 measured images). No rescaling is needed.
    radius_ratio = radius / NORMALIZED_WIDTH

    hsv_mean = sample_hsv(window, x, y, radius, valid_mask=mask)
    fruit = classify(radius_ratio, hsv_mean, max_type=SPAWN_MAX_TYPE)

    return NextResult(
        fruit=fruit,
        x=BUBBLE_X - WINDOW_HALF + x,
        y=BUBBLE_Y - WINDOW_HALF + y,
        radius=radius,
        radius_ratio=radius_ratio,
    )


def draw_debug(frame: np.ndarray, corners: np.ndarray, result: NextResult) -> None:
    label, color = _label(result)

    if result.x is None or result.y is None or result.radius is None:
        put_text(frame, label, (8, 52), color)
        return

    center, radius = screen_circle(
        inverse_warp_matrix(corners), result.x, result.y, result.radius
    )

    cv2.circle(frame, center, radius, color, 2)
    cv2.circle(frame, center, 2, color, -1)

    origin = (center[0] - radius, max(12, center[1] - radius - 6))
    put_text(frame, label, origin, color, scale=0.45, thickness=1)


def _label(result: NextResult) -> tuple[str, Color]:
    if result.fruit is not None:
        return f"next: {result.fruit.name} {result.fruit.confidence:.0f}%", (255, 0, 255)
    if result.radius_ratio is not None:
        return f"next: --- r={result.radius_ratio:.3f}", (0, 165, 255)
    return "next: ---", (0, 0, 255)


def _warp_window(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the area around the bubble at the same scale as the board.

    Measuring by dividing by the board width goes wrong when the view swings. The board appears near the center of the screen
    and the bubble near the edge, so the amount each is stretched by the projection differs.
    Warping with the same projection as the board also undoes the stretch at the bubble's position.
    """
    return warp_window(
        frame,
        corners,
        BUBBLE_X - WINDOW_HALF,
        BUBBLE_Y - WINDOW_HALF,
        WINDOW_HALF * 2,
        WINDOW_HALF * 2,
    )


def _window_mask(window: np.ndarray) -> np.ndarray:
    """Keep only the fruit inside the bubble.

    The orb and stars around the bubble are pale and drop out on saturation, but the night sky seen through the orb
    is only dark and highly saturated. Cutting by saturation alone leaves the whole bubble, and what gets measured
    is the size of the bubble instead of the fruit inside.
    """
    return solid_mask(vivid_mask(cv2.cvtColor(window, cv2.COLOR_BGR2HSV)))


def _find_blob(mask: np.ndarray) -> tuple[float, float, float] | None:
    # next only produces cherry-orange, so sizes outside that range
    # can be judged as just picking up the orb or the background.
    ratios = fruit_radius_ratios()
    min_radius = max(2.0, NORMALIZED_WIDTH * ratios[0] * 0.6)
    max_radius = max(min_radius + 1.0, NORMALIZED_WIDTH * ratios[SPAWN_MAX_TYPE] * 1.4)

    centered = [
        peak
        for peak in circle_peaks(mask, min_radius, max_radius)
        if math.hypot(peak[0] - WINDOW_HALF, peak[1] - WINDOW_HALF) <= CENTER_TOLERANCE
    ]
    if not centered:
        return None

    return max(centered, key=lambda peak: peak[2])
