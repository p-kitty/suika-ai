import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import load
from .colors import (
    COLOR_FAMILIES,
    DEFAULT_WATERMELON_RATIO,
    FRUIT_NAMES,
    FRUIT_RELATIVE_RADIUS,
    NEXT_MAX_TYPE,
    color_family,
)

__all__ = [
    "ClassifyResult",
    "classify",
    "fruit_radius_ratios",
    "sample_hsv",
    "NEXT_MAX_TYPE",
]

# 隣接段階は半径比で約 1.2 倍差。log 距離でこれを超えたら候補外にする。
RADIUS_LOG_TOLERANCE = math.log(1.6)

# 色が合わない段階への減点。候補から外さないのが重要で、色を読み違えても
# 半径がはっきりしていれば正解が残る。半径が拮抗したときだけ色が効く強さ。
COLOR_MISMATCH_PENALTY = 0.75


@dataclass
class ClassifyResult:
    type: int
    confidence: float

    @property
    def name(self) -> str:
        return FRUIT_NAMES[self.type]


def fruit_radius_ratios() -> list[float]:
    anchor = load().get("watermelon_radius_ratio", DEFAULT_WATERMELON_RATIO)
    return [relative * anchor for relative in FRUIT_RELATIVE_RADIUS]


def classify(
    radius_ratio: float,
    hsv_mean: np.ndarray | None,
    max_type: int | None = None,
) -> ClassifyResult | None:
    cfg = load()
    ratios = fruit_radius_ratios()
    upper = max_type + 1 if max_type is not None else len(ratios)

    preferred = _preferred_types(cfg, hsv_mean)

    best_type = 0
    best_score = -1.0

    for fruit_type in range(upper):
        score = _score(radius_ratio, ratios[fruit_type])
        if preferred is not None and fruit_type not in preferred:
            score *= COLOR_MISMATCH_PENALTY

        if score > best_score:
            best_score = score
            best_type = fruit_type

    if best_score < cfg.get("min_confidence", 0.35):
        return None

    return ClassifyResult(type=best_type, confidence=min(best_score, 1.0) * 100)


def sample_hsv(
    image: np.ndarray,
    x: float,
    y: float,
    radius: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    height, width = image.shape[:2]
    cx, cy = int(x), int(y)
    sample_radius = max(3, int(radius * 0.35))

    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), sample_radius, 255, -1)

    if valid_mask is not None:
        mask[valid_mask == 0] = 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 255]
    if len(pixels) == 0:
        return None

    return np.median(pixels, axis=0)


def _score(radius_ratio: float, center: float) -> float:
    if radius_ratio <= 0 or center <= 0:
        return 0.0

    deviation = abs(math.log(radius_ratio / center))
    return max(0.0, 1.0 - deviation / RADIUS_LOG_TOLERANCE)


def _preferred_types(cfg: dict, hsv_mean: np.ndarray | None) -> set[int] | None:
    """色から見て有力な段階。判断できないときは None (減点なし)。"""
    if not cfg.get("use_color_filter", True) or hsv_mean is None:
        return None

    family = color_family(float(hsv_mean[0]), float(hsv_mean[1]))
    preferred = COLOR_FAMILIES.get(family)

    return set(preferred) if preferred else None
