import cv2
import numpy as np

from .colors import FRUIT_HSV_RANGES, FRUIT_RADIUS_RATIO
from .state import Fruit


def detect(board: np.ndarray) -> list[Fruit]:
    if board.size == 0:
        return []

    circles = _find_circles(board)
    fruits = []

    for x, y, radius in circles:
        fruit = _classify(board, x, y, radius)
        if fruit is not None:
            fruits.append(fruit)

    return _deduplicate(fruits)


def draw_debug(board: np.ndarray, fruits: list[Fruit]) -> np.ndarray:
    output = board.copy()

    for fruit in fruits:
        center = (int(fruit.x), int(fruit.y))
        radius = int(fruit.radius)
        color = (0, 255, 0) if fruit.confidence >= 0.5 else (0, 165, 255)

        cv2.circle(output, center, radius, color, 2)
        cv2.circle(output, center, 2, color, -1)

        label = f"{fruit.name} {fruit.confidence:.0f}%"
        cv2.putText(
            output,
            label,
            (center[0] - radius, center[1] - radius - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"fruits: {len(fruits)}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"fruits: {len(fruits)}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return output


def _find_circles(board: np.ndarray) -> list[tuple[float, float, float]]:
    h, w = board.shape[:2]
    min_radius = max(8, int(w * FRUIT_RADIUS_RATIO[0][0]))
    max_radius = max(min_radius + 1, int(w * FRUIT_RADIUS_RATIO[-1][1]))

    contour_circles = _find_contour_circles(board, min_radius, max_radius)
    hough_circles = _find_hough_circles(board, min_radius, max_radius)

    return _merge_circle_candidates(contour_circles + hough_circles)


def _find_contour_circles(
    board: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> list[tuple[float, float, float]]:
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)

    wood_mask = cv2.inRange(hsv, (8, 35, 35), (28, 210, 190))
    dark_mask = hsv[:, :, 2] < 35
    fruit_mask = cv2.bitwise_not(wood_mask)
    fruit_mask[dark_mask] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        fruit_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    min_area = np.pi * min_radius * min_radius * 0.45
    max_area = np.pi * max_radius * max_radius * 1.35
    circles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > max_radius:
            continue

        circles.append((float(x), float(y), float(radius)))

    return circles


def _find_hough_circles(
    board: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> list[tuple[float, float, float]]:
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, min_radius),
        param1=80,
        param2=24,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]


def _merge_circle_candidates(
    circles: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    if not circles:
        return []

    circles = sorted(circles, key=lambda item: item[2], reverse=True)
    merged = []

    for circle in circles:
        x, y, radius = circle
        duplicate = False

        for mx, my, mr in merged:
            distance = np.hypot(x - mx, y - my)
            if distance < min(radius, mr) * 0.55:
                duplicate = True
                break

        if not duplicate:
            merged.append(circle)

    return merged


def _classify(
    board: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> Fruit | None:
    hsv_mean = _sample_hsv(board, x, y, radius)
    if hsv_mean is None:
        return None

    board_width = board.shape[1]
    radius_ratio = radius / board_width

    best_type = 0
    best_score = 0.0

    for fruit_type, ranges in enumerate(FRUIT_HSV_RANGES):
        color_score = _color_score(hsv_mean, ranges)
        radius_score = _radius_score(radius_ratio, fruit_type)
        score = color_score * 0.65 + radius_score * 0.35

        if score > best_score:
            best_score = score
            best_type = fruit_type

    if best_score < 0.25:
        return None

    return Fruit(
        type=best_type,
        x=x,
        y=y,
        radius=radius,
        confidence=min(best_score, 1.0) * 100,
    )


def _sample_hsv(
    board: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> np.ndarray | None:
    h, w = board.shape[:2]
    cx, cy = int(x), int(y)
    sample_radius = max(3, int(radius * 0.55))

    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), sample_radius, 255, -1)

    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 255]

    if len(pixels) == 0:
        return None

    return np.median(pixels, axis=0)


def _color_score(hsv_mean: np.ndarray, ranges: list[tuple]) -> float:
    h, s, v = hsv_mean
    best = 0.0

    for lower, upper in ranges:
        lower = np.array(lower, dtype=np.float32)
        upper = np.array(upper, dtype=np.float32)

        if lower[0] <= upper[0]:
            h_match = lower[0] <= h <= upper[0]
        else:
            h_match = h >= lower[0] or h <= upper[0]

        if not h_match:
            h_dist = min(abs(h - lower[0]), abs(h - upper[0]))
            h_score = max(0.0, 1.0 - h_dist / 25.0)
        else:
            h_score = 1.0

        s_score = _range_score(s, lower[1], upper[1], spread=80)
        v_score = _range_score(v, lower[2], upper[2], spread=80)
        score = h_score * 0.45 + s_score * 0.30 + v_score * 0.25
        best = max(best, score)

    return best


def _radius_score(radius_ratio: float, fruit_type: int) -> float:
    lower, upper = FRUIT_RADIUS_RATIO[fruit_type]
    center = (lower + upper) / 2
    half_width = (upper - lower) / 2 or 0.01
    distance = abs(radius_ratio - center)
    return max(0.0, 1.0 - distance / half_width)


def _range_score(value: float, lower: float, upper: float, spread: float) -> float:
    if lower <= value <= upper:
        return 1.0

    if value < lower:
        return max(0.0, 1.0 - (lower - value) / spread)

    return max(0.0, 1.0 - (value - upper) / spread)


def _deduplicate(fruits: list[Fruit]) -> list[Fruit]:
    if not fruits:
        return []

    fruits = sorted(fruits, key=lambda fruit: fruit.confidence, reverse=True)
    kept = []

    for fruit in fruits:
        duplicate = False

        for existing in kept:
            distance = np.hypot(fruit.x - existing.x, fruit.y - existing.y)
            if distance < min(fruit.radius, existing.radius) * 0.55:
                duplicate = True
                break

        if not duplicate:
            kept.append(fruit)

    return kept
