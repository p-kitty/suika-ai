from dataclasses import dataclass

import cv2
import numpy as np

from ..config import load
from .classify import ClassifyResult, classify, sample_hsv
from .colors import NEXT_MAX_TYPE
from .next_crop import crop_next_region


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

    blob = _find_blob(next_crop)
    if blob is None:
        return NextResult(fruit=None, region=region)

    x, y, radius = blob
    scale = load().get("next_radius_scale", 1.0)
    radius_ratio = (radius / board_width) * scale
    mask = _blob_mask(next_crop)
    hsv_mean = sample_hsv(next_crop, x, y, radius, valid_mask=mask)
    fruit = classify(radius_ratio, hsv_mean, max_type=NEXT_MAX_TYPE)

    return NextResult(
        fruit=fruit,
        region=region,
        radius_ratio=radius_ratio,
        blob=blob,
    )


def draw_debug(frame: np.ndarray, result: NextResult) -> np.ndarray:
    if result.fruit is not None:
        label = f"next: {result.fruit.name} {result.fruit.confidence:.0f}%"
        color = (255, 0, 255)
    elif result.radius_ratio is not None:
        label = f"next: --- r={result.radius_ratio:.3f}"
        color = (0, 165, 255)
    else:
        label = "next: ---"
        color = (0, 0, 255)

    if result.region is not None:
        x1, y1, x2, y2 = result.region
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        if result.blob is not None:
            bx, by, br = result.blob
            center = (int(x1 + bx), int(y1 + by))
            cv2.circle(frame, center, max(2, int(br)), (255, 0, 255), 2)
            cv2.circle(frame, center, 3, (255, 0, 255), -1)

        cv2.putText(
            frame,
            label,
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            label,
            (8, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame


def _blob_mask(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 50, 40), (180, 255, 255))
    mask[hsv[:, :, 2] < 30] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _find_blob(crop: np.ndarray) -> tuple[float, float, float] | None:
    for finder in (_find_blob_contour, _find_blob_hough, _find_blob_center):
        blob = finder(crop)
        if blob is not None:
            return blob
    return None


def _find_blob_contour(crop: np.ndarray) -> tuple[float, float, float] | None:
    height, width = crop.shape[:2]
    if height < 4 or width < 4:
        return None

    center_x = width / 2
    center_y = height / 2
    mask = _blob_mask(crop)

    margin_x = max(1, int(width * 0.05))
    margin_y = max(1, int(height * 0.05))
    mask[:margin_y, :] = 0
    mask[height - margin_y :, :] = 0
    mask[:, :margin_x] = 0
    mask[:, width - margin_x :] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_blob = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 2:
            continue

        distance = np.hypot(x - center_x, y - center_y)
        if distance > min(width, height) * 0.45:
            continue

        score = area - distance * 1.2
        if score > best_score:
            best_score = score
            best_blob = (float(x), float(y), float(radius))

    return best_blob


def _find_blob_hough(crop: np.ndarray) -> tuple[float, float, float] | None:
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    min_radius = max(3, int(min(width, height) * 0.06))
    max_radius = max(min_radius + 1, int(min(width, height) * 0.46))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_radius * 2,
        param1=80,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    center_x = width / 2
    center_y = height / 2
    best = min(
        circles[0],
        key=lambda circle: np.hypot(circle[0] - center_x, circle[1] - center_y),
    )
    return float(best[0]), float(best[1]), float(best[2])


def _find_blob_center(crop: np.ndarray) -> tuple[float, float, float] | None:
    height, width = crop.shape[:2]
    center_x = width / 2
    center_y = height / 2
    mask = _blob_mask(crop)

    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        return None

    distances = np.hypot(xs - center_x, ys - center_y)
    radius = float(np.percentile(distances, 88))
    if radius < 3:
        radius = float(np.sqrt(len(xs) / np.pi))
    if radius < 3:
        return None

    return center_x, center_y, radius
