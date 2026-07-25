from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchResult:
    name: str
    confidence: float


def match_best(image: np.ndarray, references: dict[str, np.ndarray]) -> MatchResult | None:
    if image.size == 0 or not references:
        return None

    best_name = ""
    best_score = -1.0

    for name, reference in references.items():
        score = _compare(image, reference)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < 0.45:
        return None

    return MatchResult(name=best_name, confidence=min(best_score, 1.0) * 100)


def _compare(image: np.ndarray, reference: np.ndarray) -> float:
    if reference.size == 0:
        return -1.0

    size = 48
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    image_gray = cv2.resize(image_gray, (size, size))
    reference_gray = cv2.resize(reference_gray, (size, size))

    result = cv2.matchTemplate(image_gray, reference_gray, cv2.TM_CCOEFF_NORMED)
    return float(result[0, 0])
