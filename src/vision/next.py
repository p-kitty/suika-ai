from dataclasses import dataclass

import cv2
import numpy as np

from ..config import load
from ..draw import Color, put_text
from .blobs import circle_peaks
from .classify import ClassifyResult, classify, fruit_radius_ratios, sample_hsv
from .colors import NEXT_MAX_TYPE, saturated_mask
from .next_crop import crop_next_region

# crop 中心からこの割合より離れたピークは next のフルーツではない。
CENTER_TOLERANCE_RATIO = 0.35


@dataclass
class NextResult:
    fruit: ClassifyResult | None
    region: tuple[int, int, int, int] | None
    radius_ratio: float | None = None
    blob: tuple[float, float, float] | None = None


def detect(frame: np.ndarray, corners: np.ndarray) -> NextResult:
    region = crop_next_region(frame, corners)
    if region is None:
        return NextResult(fruit=None, region=None)

    top_left, top_right, _bottom_right, _bottom_left = corners
    board_width = float(np.linalg.norm(top_right - top_left))
    if board_width <= 0:
        return NextResult(fruit=None, region=region)

    x1, y1, x2, y2 = region
    next_crop = frame[y1:y2, x1:x2]
    if next_crop.size == 0:
        return NextResult(fruit=None, region=region)

    mask = _blob_mask(next_crop)
    scale = load().get("next_radius_scale", 1.0) or 1.0
    blob = _find_blob(mask, board_width, scale)
    if blob is None:
        return NextResult(fruit=None, region=region)

    x, y, radius = blob
    radius_ratio = (radius / board_width) * scale
    hsv_mean = sample_hsv(next_crop, x, y, radius, valid_mask=mask)
    fruit = classify(radius_ratio, hsv_mean, max_type=NEXT_MAX_TYPE)

    return NextResult(
        fruit=fruit,
        region=region,
        radius_ratio=radius_ratio,
        blob=blob,
    )


def draw_debug(frame: np.ndarray, result: NextResult) -> None:
    label, color = _label(result)

    if result.region is None:
        put_text(frame, label, (8, 52), color)
        return

    x1, y1, x2, y2 = result.region
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    if result.blob is not None:
        bx, by, br = result.blob
        center = (int(x1 + bx), int(y1 + by))
        cv2.circle(frame, center, max(2, int(br)), (255, 0, 255), 2)
        cv2.circle(frame, center, 3, (255, 0, 255), -1)

    put_text(frame, label, (x1, max(16, y1 - 8)), color, scale=0.55)


def _label(result: NextResult) -> tuple[str, Color]:
    if result.fruit is not None:
        return f"next: {result.fruit.name} {result.fruit.confidence:.0f}%", (255, 0, 255)
    if result.radius_ratio is not None:
        return f"next: --- r={result.radius_ratio:.3f}", (0, 165, 255)
    return "next: ---", (0, 0, 255)


def _blob_mask(crop: np.ndarray) -> np.ndarray:
    """next のフルーツだけを残す。

    プレビューは彩度の低い白い玉に包まれているので、彩度で切れば
    玉ではなく中のフルーツが残る。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = saturated_mask(hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _find_blob(
    mask: np.ndarray,
    board_width: float,
    scale: float,
) -> tuple[float, float, float] | None:
    height, width = mask.shape[:2]
    if height < 4 or width < 4:
        return None

    # next は cherry〜orange しか出ないので、その範囲外の大きさは
    # 玉や背景を拾っているだけと判断できる。
    ratios = fruit_radius_ratios()
    min_radius = max(2.0, board_width * ratios[0] * 0.6 / scale)
    max_radius = max(min_radius + 1.0, board_width * ratios[NEXT_MAX_TYPE] * 1.4 / scale)

    peaks = circle_peaks(mask, min_radius, max_radius)
    if not peaks:
        return None

    center_x = width / 2
    center_y = height / 2
    tolerance = min(width, height) * CENTER_TOLERANCE_RATIO

    centered = [
        peak
        for peak in peaks
        if np.hypot(peak[0] - center_x, peak[1] - center_y) <= tolerance
    ]
    if not centered:
        return None

    return max(centered, key=lambda peak: peak[2])
