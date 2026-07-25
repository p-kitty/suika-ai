import math

import cv2
import numpy as np

from ..config import load
from .colors import FRUIT_NAMES


def crop_icons(frame: np.ndarray, corners: np.ndarray) -> dict[str, np.ndarray]:
    cfg = load().get("merge_wheel")
    if cfg is None:
        return {}

    _top_left, top_right, _bottom_right, bottom_left = corners
    board_width = float(np.linalg.norm(top_right - _top_left))
    board_height = float(np.linalg.norm(bottom_left - _top_left))

    if board_width <= 0 or board_height <= 0:
        return {}

    center_x = top_right[0] + cfg["offset_x_ratio"] * board_width
    center_y = top_right[1] + cfg["offset_y_ratio"] * board_height
    radius = cfg["radius_ratio"] * board_width
    icon_half = cfg["icon_size_ratio"] * board_width / 2

    icons = {}
    height, width = frame.shape[:2]

    for index, name in enumerate(FRUIT_NAMES):
        angle = -math.pi / 2 + index * (2 * math.pi / len(FRUIT_NAMES))
        icon_x = center_x + radius * math.cos(angle)
        icon_y = center_y + radius * math.sin(angle)

        x1 = int(round(icon_x - icon_half))
        y1 = int(round(icon_y - icon_half))
        x2 = int(round(icon_x + icon_half))
        y2 = int(round(icon_y + icon_half))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            icons[name] = crop

    return icons
