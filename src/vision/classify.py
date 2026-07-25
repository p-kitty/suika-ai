from dataclasses import dataclass

import cv2
import numpy as np

from .colors import COLOR_FAMILIES, FRUIT_NAMES, FRUIT_RADIUS_RATIO, NEXT_MAX_TYPE, color_family

__all__ = ["ClassifyResult", "classify", "radius_score", "sample_hsv", "NEXT_MAX_TYPE"]


@dataclass
class ClassifyResult:
    type: int
    confidence: float

    @property
    def name(self) -> str:
        return FRUIT_NAMES[self.type]


def classify(
    radius_ratio: float,
    hsv_mean: np.ndarray | None,
    max_type: int | None = None,
) -> ClassifyResult | None:
    if hsv_mean is None:
        family = "red_orange"
    else:
        family = color_family(float(hsv_mean[0]), float(hsv_mean[1]))

    pool = _candidate_pool(family, max_type)
    candidates = _radius_candidates(radius_ratio, pool)

    best_type = candidates[0]
    best_score = radius_score(radius_ratio, best_type)

    for fruit_type in candidates[1:]:
        score = radius_score(radius_ratio, fruit_type)
        if score > best_score:
            best_score = score
            best_type = fruit_type

    if best_score < 0.45:
        return None

    return ClassifyResult(type=best_type, confidence=min(best_score, 1.0) * 100)


def radius_score(radius_ratio: float, fruit_type: int) -> float:
    lower, upper = FRUIT_RADIUS_RATIO[fruit_type]
    center = (lower + upper) / 2
    half_width = (upper - lower) / 2 or 0.01
    distance = abs(radius_ratio - center)
    return max(0.0, 1.0 - distance / half_width)


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


def _candidate_pool(family: str, max_type: int | None) -> list[int]:
    upper = max_type + 1 if max_type is not None else len(FRUIT_RADIUS_RATIO)
    full_pool = list(range(upper))

    if family == "unknown":
        return full_pool

    family_pool = COLOR_FAMILIES.get(family, full_pool)
    pool = [fruit_type for fruit_type in family_pool if fruit_type < upper]
    return pool if pool else full_pool


def _radius_candidates(radius_ratio: float, pool: list[int]) -> list[int]:
    scored = [(radius_score(radius_ratio, fruit_type), fruit_type) for fruit_type in pool]
    scored.sort(reverse=True)

    candidates = [fruit_type for score, fruit_type in scored if score > 0.2][:3]
    if not candidates:
        candidates = [scored[0][1]]

    return candidates
