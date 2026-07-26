import cv2
import numpy as np

from .blobs import circle_peaks
from .classify import classify, fruit_radius_ratios, sample_hsv
from .colors import BOARD_BG_HSV, saturated_mask
from .state import Fruit

# The detected corners are outside the frame, so the warped board shows the frame and the shadow inside it.
# Dropping that band makes the boundary of fruits touching the walls correctly the inner wall.
BORDER_BAND_RATIO = 0.045

# The background is a gradient getting smoothly darker from top to bottom, and its lower part is about as saturated
# as saturated. A fixed threshold cannot cut it, so the background is found by region growing
# from the open top.
BACKGROUND_SEED_BAND = 0.30
BACKGROUND_SEED_STEP = 16
# Judged by the difference from adjacent pixels, so it follows gentle gradients while stopping at contours.
BACKGROUND_TOLERANCE = 10
# Points far from the representative background color are not used as seeds (does not break even with fruit at the top).
BACKGROUND_SEED_TOLERANCE = 26
# If the background gets no more than this, region growing is considered failed.
MIN_BACKGROUND_RATIO = 0.20

# The background saturation overlaps with fruits, so a threshold alone cannot separate them.
# Keep only 'balls actually visible' by how much contour lies on the circumference.
EDGE_SUPPORT_SAMPLES = 48
# Overlapping fruits have part of their circumference hidden, so it is set low.
MIN_EDGE_SUPPORT = 0.25
EDGE_GRADIENT_THRESHOLD = 60.0


def detect(board: np.ndarray) -> list[Fruit]:
    if board.size == 0:
        return []

    mask = fruit_mask(board)
    outline = _outline(board)
    fruits = []

    for x, y, radius in _find_circles(board.shape[1], mask):
        if _edge_support(outline, x, y, radius) < MIN_EDGE_SUPPORT:
            continue

        fruit = _classify(board, mask, x, y, radius)
        if fruit is not None:
            fruits.append(fruit)

    return _deduplicate(fruits)


def _outline(board: np.ndarray) -> np.ndarray:
    """The contours actually visible in the image.

    The mask boundary is not used. A blob made by the threshold picking up background always has its own boundary,
    so including it in the contours makes it indistinguishable from a real ball.

    Looking only at lightness misses the contours of fruits close to the background in brightness but different in color only
    (a pear on beige and so on). Gradients are taken over all Lab channels.
    """
    lab = cv2.GaussianBlur(cv2.cvtColor(board, cv2.COLOR_BGR2LAB), (5, 5), 0)

    gradient = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel in range(3):
        plane = lab[:, :, channel]
        gx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.maximum(gradient, cv2.magnitude(gx, gy))

    edges = (gradient >= EDGE_GRADIENT_THRESHOLD).astype(np.uint8) * 255

    return cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def _edge_support(outline: np.ndarray, x: float, y: float, radius: float) -> float:
    height, width = outline.shape[:2]
    angles = np.linspace(0.0, 2.0 * np.pi, EDGE_SUPPORT_SAMPLES, endpoint=False)

    xs = np.clip((x + radius * np.cos(angles)).astype(int), 0, width - 1)
    ys = np.clip((y + radius * np.sin(angles)).astype(int), 0, height - 1)

    return float((outline[ys, xs] > 0).mean())


def fruit_mask(board: np.ndarray) -> np.ndarray:
    background = _background_mask(board)

    if float((background > 0).mean()) < MIN_BACKGROUND_RATIO:
        mask = _saturation_mask(board)
    else:
        mask = np.where(background > 0, 0, 255).astype(np.uint8)

    # The frame, shadows and the background outside the frame are all highly saturated and cannot be cut by color,
    # so the board's edges are dropped wholesale without looking at color.
    mask[_border_band(mask.shape)] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def _background_mask(board: np.ndarray) -> np.ndarray:
    """Flood the background from the top where no fruit is stacked.

    Calling floodFill without FIXED_RANGE compares not against 'the seed color' but against
    'the neighboring already filled pixel'. So a gentle gradient from top to bottom can be followed
    to the end, and it stops only at sharp steps like fruit edges.
    """
    height, width = board.shape[:2]
    # A Gaussian softens edges, and raising the tolerance even slightly makes the fill
    # leak into fruits. A median preserves steps, so it is insensitive to tolerance.
    blurred = cv2.medianBlur(board, 5)

    margin_y = max(1, int(height * BORDER_BAND_RATIO))
    margin_x = max(1, int(width * BORDER_BAND_RATIO))
    band_bottom = max(margin_y + 1, int(height * BACKGROUND_SEED_BAND))

    band = blurred[margin_y:band_bottom, margin_x : width - margin_x]
    if band.size == 0:
        return np.zeros((height, width), dtype=np.uint8)

    reference = np.median(band.reshape(-1, 3), axis=0)

    filled = np.zeros((height + 2, width + 2), dtype=np.uint8)
    tolerance = (BACKGROUND_TOLERANCE,) * 3
    flags = cv2.FLOODFILL_MASK_ONLY | 4 | (255 << 8)

    for y in range(margin_y, band_bottom, BACKGROUND_SEED_STEP):
        for x in range(margin_x, width - margin_x, BACKGROUND_SEED_STEP):
            if filled[y + 1, x + 1]:
                continue
            if np.abs(blurred[y, x].astype(np.int16) - reference).max() > BACKGROUND_SEED_TOLERANCE:
                continue

            cv2.floodFill(blurred, filled, (x, y), 0, tolerance, tolerance, flags)

    return filled[1:-1, 1:-1]


def _saturation_mask(board: np.ndarray) -> np.ndarray:
    """A fallback for when region growing fails."""
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    mask = saturated_mask(hsv)

    bg_lower, bg_upper = BOARD_BG_HSV
    mask[cv2.inRange(hsv, bg_lower, bg_upper) > 0] = 0

    return mask


def _border_band(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    band = np.zeros(shape, dtype=bool)

    margin_y = max(1, int(height * BORDER_BAND_RATIO))
    margin_x = max(1, int(width * BORDER_BAND_RATIO))

    band[:margin_y, :] = True
    band[height - margin_y :, :] = True
    band[:, :margin_x] = True
    band[:, width - margin_x :] = True

    return band


def _find_circles(
    board_width: int,
    mask: np.ndarray,
) -> list[tuple[float, float, float]]:
    ratios = fruit_radius_ratios()
    min_radius = max(3.0, board_width * ratios[0] * 0.7)
    max_radius = max(min_radius + 2.0, board_width * ratios[-1] * 1.3)

    return circle_peaks(mask, min_radius, max_radius)


def _classify(
    board: np.ndarray,
    mask: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> Fruit | None:
    radius_ratio = radius / board.shape[1]

    hsv_mean = sample_hsv(board, x, y, radius, valid_mask=mask)
    result = classify(radius_ratio, hsv_mean)

    if result is None:
        return None

    return Fruit(
        type=result.type,
        x=x,
        y=y,
        radius=radius,
        confidence=result.confidence,
    )


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
