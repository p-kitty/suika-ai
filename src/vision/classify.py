import math
from dataclasses import dataclass

import cv2
import numpy as np

from .colors import (
    COLOR_FAMILIES,
    FRUIT_NAMES,
    FRUIT_RELATIVE_RADIUS,
    WATERMELON_RADIUS_RATIO,
    color_family,
)
from .normalized import NORMALIZED_WIDTH

# Adjacent stages differ by about 1.2x in radius ratio. Beyond this in log distance, a candidate is excluded.
RADIUS_LOG_TOLERANCE = math.log(1.6)

# Penalty for stages whose color does not match. Not excluding them is key: even if the color is misread,
# the correct answer remains when the radius is clear. Strong enough to matter only when radii are close.
COLOR_MISMATCH_PENALTY = 0.75
MIN_CONFIDENCE = 0.35
USE_COLOR_FILTER = True


@dataclass
class ClassifyResult:
    type: int
    confidence: float

    @property
    def name(self) -> str:
        return FRUIT_NAMES[self.type]


def fruit_radius_ratios() -> list[float]:
    return [relative * WATERMELON_RADIUS_RATIO for relative in FRUIT_RELATIVE_RADIUS]


def fruit_radius(fruit_type: int) -> float:
    """Radius on the normalized board."""
    return fruit_radius_ratios()[fruit_type] * NORMALIZED_WIDTH


def classify(
    radius_ratio: float,
    hsv_mean: np.ndarray | None,
    max_type: int | None = None,
) -> ClassifyResult | None:
    ratios = fruit_radius_ratios()
    upper = max_type + 1 if max_type is not None else len(ratios)

    preferred = _preferred_types(hsv_mean)

    best_type = 0
    best_score = -1.0

    for fruit_type in range(upper):
        score = _score(radius_ratio, ratios[fruit_type])
        if preferred is not None and fruit_type not in preferred:
            score *= COLOR_MISMATCH_PENALTY

        if score > best_score:
            best_score = score
            best_type = fruit_type

    if best_score < MIN_CONFIDENCE:
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


def _preferred_types(hsv_mean: np.ndarray | None) -> set[int] | None:
    """The stage suggested by color. None when undecidable (no penalty)."""
    if not USE_COLOR_FILTER or hsv_mean is None:
        return None

    family = color_family(float(hsv_mean[0]), float(hsv_mean[1]))
    preferred = COLOR_FAMILIES.get(family)

    return set(preferred) if preferred else None
