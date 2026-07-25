from dataclasses import dataclass

import cv2
import numpy as np

from .match import MatchResult, match_best
from .next_crop import crop_next_region
from .wheel import crop_icons


@dataclass
class NextResult:
    fruit: MatchResult | None
    region: tuple[int, int, int, int] | None


def detect(frame: np.ndarray, corners: np.ndarray) -> NextResult:
    region = crop_next_region(frame, corners)
    if region is None:
        return NextResult(fruit=None, region=None)

    x1, y1, x2, y2 = region
    next_crop = frame[y1:y2, x1:x2]
    wheel_icons = crop_icons(frame, corners)
    fruit = match_best(next_crop, wheel_icons)

    return NextResult(fruit=fruit, region=region)


def draw_debug(frame: np.ndarray, result: NextResult) -> np.ndarray:
    output = frame.copy()

    if result.region is not None:
        x1, y1, x2, y2 = result.region
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 200, 0), 2)

    if result.fruit is not None:
        label = f"next: {result.fruit.name} {result.fruit.confidence:.0f}%"
        color = (255, 200, 0)
    else:
        label = "next: ---"
        color = (0, 0, 255)

    cv2.putText(
        output,
        label,
        (8, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    return output
