from dataclasses import dataclass

import cv2
import numpy as np

from .next import NextResult, detect as detect_next, draw_debug as draw_next_debug
from .wheel import draw_debug as draw_wheel_debug

NORMALIZED_WIDTH = 400
NORMALIZED_HEIGHT = 500

YELLOW_HSV_LOWER = (18, 60, 100)
YELLOW_HSV_UPPER = (45, 255, 255)

MIN_BOX_AREA_RATIO = 0.02


@dataclass
class BoardResult:
    normalized: np.ndarray | None
    corners: np.ndarray | None
    found: bool
    next_fruit: NextResult | None = None


def localize(frame: np.ndarray) -> BoardResult:
    corners = _find_corners(frame)
    if corners is None:
        return BoardResult(normalized=None, corners=None, found=False)

    normalized = _warp(frame, corners)
    next_fruit = detect_next(frame, corners)
    return BoardResult(
        normalized=normalized,
        corners=corners,
        found=True,
        next_fruit=next_fruit,
    )


def draw_frame_debug(frame: np.ndarray, result: BoardResult) -> np.ndarray:
    output = frame.copy()

    if result.found and result.corners is not None:
        corners = result.corners.astype(int)
        cv2.polylines(output, [corners], True, (0, 255, 0), 2)

        for i, (x, y) in enumerate(corners):
            cv2.circle(output, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                output,
                str(i),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            output,
            "board: detected",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        output = draw_wheel_debug(output, result.corners)

        if result.next_fruit is not None:
            output = draw_next_debug(output, result.next_fruit)
    else:
        cv2.putText(
            output,
            "board: not found",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def _find_corners(frame: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * MIN_BOX_AREA_RATIO

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        corners = _contour_to_corners(contour)
        if corners is not None:
            candidates.append((area, corners))

    if not candidates:
        return None

    _, corners = max(candidates, key=lambda item: item[0])
    return _order_corners(corners)


def _contour_to_corners(contour: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)


def _order_corners(corners: np.ndarray) -> np.ndarray:
    corners = np.array(corners, dtype=np.float32)
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).reshape(-1)

    top_left = corners[np.argmin(sums)]
    bottom_right = corners[np.argmax(sums)]
    top_right = corners[np.argmin(diffs)]
    bottom_left = corners[np.argmax(diffs)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _warp(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    destination = np.array(
        [
            [0, 0],
            [NORMALIZED_WIDTH - 1, 0],
            [NORMALIZED_WIDTH - 1, NORMALIZED_HEIGHT - 1],
            [0, NORMALIZED_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(
        frame,
        matrix,
        (NORMALIZED_WIDTH, NORMALIZED_HEIGHT),
    )
