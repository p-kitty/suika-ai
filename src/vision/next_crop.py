import numpy as np

from ..config import load


def crop_next_region(
    frame: np.ndarray,
    corners: np.ndarray,
) -> tuple[int, int, int, int] | None:
    cfg = load().get("next_crop")
    if cfg is None:
        return None

    top_left, top_right, _bottom_right, bottom_left = corners
    board_width = float(np.linalg.norm(top_right - top_left))
    board_height = float(np.linalg.norm(bottom_left - top_left))

    if board_width <= 0 or board_height <= 0:
        return None

    center_x = top_right[0] + cfg["offset_x_ratio"] * board_width
    center_y = top_right[1] + cfg["offset_y_ratio"] * board_height
    half_size = cfg["size_ratio"] * board_width / 2

    x1 = int(round(center_x - half_size))
    y1 = int(round(center_y - half_size))
    x2 = int(round(center_x + half_size))
    y2 = int(round(center_y + half_size))

    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return x1, y1, x2, y2
