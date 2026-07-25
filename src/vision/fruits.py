import cv2
import numpy as np

from .colors import BOARD_BG_HSV, FRUIT_RADIUS_RATIO, color_family
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
            (center[0] - radius, max(12, center[1] - radius - 6)),
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


def _fruit_mask(board: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)

    bg_lower, bg_upper = BOARD_BG_HSV
    bg_mask = cv2.inRange(hsv, bg_lower, bg_upper)

    fruit_mask = cv2.inRange(hsv, (0, 90, 55), (180, 255, 255))
    fruit_mask[bg_mask > 0] = 0
    fruit_mask[hsv[:, :, 2] < 40] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)

    return fruit_mask


def _edge_image(board: np.ndarray, fruit_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges[fruit_mask == 0] = 0
    return edges


def _find_circles(board: np.ndarray) -> list[tuple[float, float, float]]:
    w = board.shape[1]
    min_radius = max(8, int(w * FRUIT_RADIUS_RATIO[0][0] * 0.85))
    max_radius = max(min_radius + 1, int(w * FRUIT_RADIUS_RATIO[-1][1]))

    fruit_mask = _fruit_mask(board)
    edges = _edge_image(board, fruit_mask)

    circles = []
    circles.extend(_hough_circles(edges, min_dist=45, min_radius=25, max_radius=max_radius, param2=40))
    circles.extend(_hough_circles(edges, min_dist=28, min_radius=min_radius, max_radius=24, param2=35))
    circles.extend(_small_contour_circles(fruit_mask, min_radius, 26))

    return _merge_circle_candidates(circles)


def _hough_circles(
    edges: np.ndarray,
    min_dist: int,
    min_radius: int,
    max_radius: int,
    param2: int,
) -> list[tuple[float, float, float]]:
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=100,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]


def _small_contour_circles(
    fruit_mask: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> list[tuple[float, float, float]]:
    contours, _ = cv2.findContours(
        fruit_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    circles = []
    min_area = np.pi * min_radius * min_radius * 0.45

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.55:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > max_radius:
            continue

        circles.append((float(x), float(y), float(radius)))

    return circles


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
            if distance < max(radius, mr) * 0.60:
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
    board_width = board.shape[1]
    radius_ratio = radius / board_width

    hsv_mean = _sample_hsv(board, x, y, radius)
    family = "red_orange" if hsv_mean is None else color_family(float(hsv_mean[0]), float(hsv_mean[1]))

    candidates = _radius_candidates(radius_ratio, family)

    best_type = candidates[0]
    best_score = _radius_score(radius_ratio, best_type)

    for fruit_type in candidates[1:]:
        score = _radius_score(radius_ratio, fruit_type)
        if score > best_score:
            best_score = score
            best_type = fruit_type

    if best_score < 0.45:
        return None

    return Fruit(
        type=best_type,
        x=x,
        y=y,
        radius=radius,
        confidence=min(best_score, 1.0) * 100,
    )


def _radius_candidates(radius_ratio: float, family: str) -> list[int]:
    from .colors import COLOR_FAMILIES

    if family == "unknown":
        pool = list(range(len(FRUIT_RADIUS_RATIO)))
    else:
        pool = COLOR_FAMILIES.get(family, list(range(len(FRUIT_RADIUS_RATIO))))

    scored = []
    for fruit_type in pool:
        score = _radius_score(radius_ratio, fruit_type)
        scored.append((score, fruit_type))

    scored.sort(reverse=True)
    candidates = [fruit_type for score, fruit_type in scored if score > 0.2][:3]

    if not candidates:
        candidates = [scored[0][1]]

    return candidates


def _sample_hsv(
    board: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> np.ndarray | None:
    h, w = board.shape[:2]
    cx, cy = int(x), int(y)
    sample_radius = max(3, int(radius * 0.35))

    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), sample_radius, 255, -1)

    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    fruit_mask = _fruit_mask(board)
    mask[fruit_mask == 0] = 0

    pixels = hsv[mask == 255]
    if len(pixels) == 0:
        return None

    return np.median(pixels, axis=0)


def _radius_score(radius_ratio: float, fruit_type: int) -> float:
    lower, upper = FRUIT_RADIUS_RATIO[fruit_type]
    center = (lower + upper) / 2
    half_width = (upper - lower) / 2 or 0.01
    distance = abs(radius_ratio - center)
    return max(0.0, 1.0 - distance / half_width)


def _deduplicate(fruits: list[Fruit]) -> list[Fruit]:
    if not fruits:
        return []

    fruits = sorted(fruits, key=lambda fruit: (fruit.radius, fruit.confidence), reverse=True)
    kept = []

    for fruit in fruits:
        duplicate = False

        for existing in kept:
            distance = np.hypot(fruit.x - existing.x, fruit.y - existing.y)
            overlap = fruit.radius + existing.radius - distance
            if overlap > min(fruit.radius, existing.radius) * 0.50:
                duplicate = True
                break

        if not duplicate:
            kept.append(fruit)

    return kept
