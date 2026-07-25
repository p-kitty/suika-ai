import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import load
from .colors import FRUIT_NAMES


@dataclass
class WheelIcon:
    name: str
    region: tuple[int, int, int, int]


@dataclass
class WheelGeometry:
    center: tuple[int, int]
    radius: int
    icons: list[WheelIcon]


def _geometry(corners: np.ndarray) -> WheelGeometry | None:
    cfg = load().get("merge_wheel")
    if cfg is None:
        return None

    _top_left, top_right, _bottom_right, bottom_left = corners
    board_width = float(np.linalg.norm(top_right - _top_left))
    board_height = float(np.linalg.norm(bottom_left - _top_left))

    if board_width <= 0 or board_height <= 0:
        return None

    center_x = top_right[0] + cfg["offset_x_ratio"] * board_width
    center_y = top_right[1] + cfg["offset_y_ratio"] * board_height
    radius = cfg["radius_ratio"] * board_width
    icon_half = cfg["icon_size_ratio"] * board_width / 2
    start_angle = -math.pi / 2 + cfg.get("start_angle_offset", 0.0)

    icons = []
    for index, name in enumerate(FRUIT_NAMES):
        angle = start_angle + index * (2 * math.pi / len(FRUIT_NAMES))
        icon_x = center_x + radius * math.cos(angle)
        icon_y = center_y + radius * math.sin(angle)

        x1 = int(round(icon_x - icon_half))
        y1 = int(round(icon_y - icon_half))
        x2 = int(round(icon_x + icon_half))
        y2 = int(round(icon_y + icon_half))
        icons.append(WheelIcon(name=name, region=(x1, y1, x2, y2)))

    return WheelGeometry(
        center=(int(round(center_x)), int(round(center_y))),
        radius=int(round(radius)),
        icons=icons,
    )


def crop_icons(frame: np.ndarray, corners: np.ndarray) -> dict[str, np.ndarray]:
    geometry = _geometry(corners)
    if geometry is None:
        return {}

    height, width = frame.shape[:2]
    icons = {}

    for icon in geometry.icons:
        x1, y1, x2, y2 = icon.region
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            icons[icon.name] = crop

    return icons


def draw_debug(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    output = frame.copy()
    geometry = _geometry(corners)
    if geometry is None:
        return output

    height, width = frame.shape[:2]
    color = (255, 0, 255)

    cv2.circle(output, geometry.center, 4, color, -1)
    cv2.circle(output, geometry.center, geometry.radius, color, 1)

    for icon in geometry.icons:
        x1, y1, x2, y2 = icon.region
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            output,
            icon.name[:3],
            (x1, max(y1 - 2, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    return output
