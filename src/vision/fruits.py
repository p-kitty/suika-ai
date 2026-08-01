from functools import cache

import cv2
import numpy as np

from .blobs import border_labels, circle_peaks
from .classify import classify, fruit_radius_ratios, sample_hsv
from .colors import BOARD_BG_HSV, saturated_mask
from .state import Fruit

# The detected corners are outside the frame, so the warped board shows the frame and the shadow inside it.
# Dropping that band makes the boundary of fruits touching the walls correctly the inner wall.
BORDER_BAND_RATIO = 0.045

# The background is a gradient getting smoothly darker from top to bottom, and its lower part is about as saturated
# as the fruits. A fixed threshold cannot cut it, so the background color is fitted as a linear function of coordinates
# and fruits are taken by color difference from it.
BACKGROUND_TOLERANCE = 14.0
BACKGROUND_FIT_ITERATIONS = 5
# Determining the plane does not need every pixel. Fit on a thinned sample.
BACKGROUND_FIT_STRIDE = 4
# Pixel count after thinning. If the seeds are filled with fruit, the fit is considered failed.
MIN_BACKGROUND_SAMPLES = 300
# Width of the ring the seeds are taken from. Taken just inside the edge band.
BACKGROUND_SEED_WIDTH_RATIO = 0.03

# The background saturation overlaps with fruits, so a threshold alone cannot separate them.
# Keep only 'balls actually visible' by how much contour lies on the circumference.
EDGE_SUPPORT_SAMPLES = 48
# Overlapping fruits have part of their circumference hidden, so it is set low.
MIN_EDGE_SUPPORT = 0.25
EDGE_GRADIENT_THRESHOLD = 60.0

# Distance from the background color for treating an enclosed hole as a highlight. Measured, highlights are
# 12-20 and real background enclosed by touching fruits is 1-4, with a gap between.
HOLE_BACKGROUND_DISTANCE = 8.0


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
    distance = _background_distance(board)

    if distance is None:
        mask = _saturation_mask(board)
    else:
        mask = (distance > BACKGROUND_TOLERANCE).astype(np.uint8) * 255

    # The frame, shadows and the background outside the frame are all highly saturated and cannot be cut by color,
    # so the board's edges are dropped wholesale without looking at color.
    height, width = (int(v) for v in mask.shape[:2])
    mask[_border_band((height, width))] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if distance is None:
        return mask

    return _fill_highlights(mask, distance)


def _fill_highlights(mask: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Fill holes left by highlights on fruit surfaces.

    Strong highlights appear close to the beige background color and fall within tolerance,
    leaving holes inside the fruit. A hole pushes the distance transform's peak away from the center,
    and one fruit splits into several small circles.

    Background enclosed by touching fruits is also an enclosed hole, but filling that
    joins the fruits into one huge circle. They are told apart by the distance from the background color:
    highlights sit at the edge of tolerance, while real background is well inside.
    """
    background = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)

    # Label 0 is the fruit itself. Regions continuing outside the board are background and left alone.
    enclosed = np.ones(count, dtype=bool)
    enclosed[0] = False
    enclosed[border_labels(labels)] = False

    highlight = np.zeros(count, dtype=bool)
    for label in np.nonzero(enclosed)[0]:
        left, top, width, height = stats[label, :4]
        window = (slice(top, top + height), slice(left, left + width))
        hole = labels[window] == label

        highlight[label] = np.median(distance[window][hole]) >= HOLE_BACKGROUND_DISTANCE

    filled = mask.copy()
    filled[highlight[labels]] = 255

    return filled


def _background_distance(board: np.ndarray) -> np.ndarray | None:
    """Return how far each pixel is from the background color.

    The background only gets smoothly darker from top to bottom, so the color, as a linear function of coordinates
    can be written as Lab = a + b*x + c*y. Fruits are repeatedly cut off as outliers,
    and the fit uses only the remaining pixels.

    Unlike flood filling it does not break when the board is full, and background isolated
    by surrounding fruits is still picked up as background.
    """
    # A median preserves steps, so edges do not soften and the color difference stays sharp.
    lab = cv2.cvtColor(cv2.medianBlur(board, 5), cv2.COLOR_BGR2LAB).astype(np.float32)

    coefficients = _fit_background(lab, _seed_band(board.shape[:2]))
    if coefficients is None:
        return None

    # With only the edge band, the seeds lean toward the outer rim of the screen. Refitting with pixels judged as background
    # by the first fit makes it decided by seeds spread over the whole board.
    inliers = _distance_to(lab, coefficients) < BACKGROUND_TOLERANCE
    refitted = _fit_background(lab, inliers)

    return _distance_to(lab, coefficients if refitted is None else refitted)


def _fit_background(lab: np.ndarray, seeds: np.ndarray) -> np.ndarray | None:
    """Find the linear coefficients (3x3) while cutting off outliers."""
    stride = BACKGROUND_FIT_STRIDE
    design = _coordinate_design(lab.shape[:2])
    samples = lab[::stride, ::stride].reshape(-1, 3)
    keep = seeds[::stride, ::stride].reshape(-1).copy()

    for _ in range(BACKGROUND_FIT_ITERATIONS):
        if int(keep.sum()) < MIN_BACKGROUND_SAMPLES:
            return None

        rows = design[keep]
        coefficients = np.linalg.lstsq(rows, samples[keep], rcond=None)[0]
        residual = np.linalg.norm(rows @ coefficients - samples[keep], axis=1)

        outliers = residual >= BACKGROUND_TOLERANCE
        if not outliers.any():
            break

        keep[np.nonzero(keep)[0][outliers]] = False

    return coefficients


def _distance_to(lab: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """How far each pixel is from the background color given by the linear function."""
    height, width = lab.shape[:2]
    constant, per_x, per_y = coefficients

    model = constant + np.arange(width, dtype=np.float32)[None, :, None] * per_x
    model = model + np.arange(height, dtype=np.float32)[:, None, None] * per_y

    return np.linalg.norm(lab - model, axis=2)


@cache
def _coordinate_design(shape: tuple[int, int]) -> np.ndarray:
    """Thinned coordinates [1, x, y] for fitting. Determining the plane does not need every pixel."""
    height, width = shape
    stride = BACKGROUND_FIT_STRIDE
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]

    return np.stack([np.ones_like(xs), xs, ys], axis=-1).astype(np.float32).reshape(-1, 3)


def _seed_band(shape: tuple[int, int]) -> np.ndarray:
    """A ring just inside the edge band. Background is often visible near the walls."""
    outer = _border_band(shape)
    inner = _border_band(shape, BORDER_BAND_RATIO + BACKGROUND_SEED_WIDTH_RATIO)

    return inner & ~outer


def _saturation_mask(board: np.ndarray) -> np.ndarray:
    """A fallback for when fitting fails."""
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    mask = saturated_mask(hsv)

    bg_lower, bg_upper = BOARD_BG_HSV
    mask[cv2.inRange(hsv, bg_lower, bg_upper) > 0] = 0

    return mask


def _border_band(shape: tuple[int, int], ratio: float = BORDER_BAND_RATIO) -> np.ndarray:
    height, width = shape
    band = np.zeros(shape, dtype=bool)

    margin_y = max(1, int(height * ratio))
    margin_x = max(1, int(width * ratio))

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
            # Only a second peak standing on the same fruit should be dropped. A different touching
            # fruit does not bite into the center even if they overlap somewhat visually.
            if distance < max(fruit.radius, existing.radius):
                duplicate = True
                break

        if not duplicate:
            kept.append(fruit)

    return kept
