import cv2
import numpy as np

MAX_PEAK_CANDIDATES = 400
# Peaks closer than this ratio are considered the same circle.
NMS_RATIO = 0.65

# Range for checking whether a peak is a summit. A ratio of its own radius.
APEX_REACH_RATIO = 0.7
# The distance transform is approximate, so even a flat top has slightly varying heights. Measured, real ones
# stay within 1.001x, peaks on ridges are 1.13x or more, and there is a gap between.
APEX_TOLERANCE = 1.05


def circle_peaks(
    mask: np.ndarray,
    min_radius: float,
    max_radius: float,
) -> list[tuple[float, float, float]]:
    """Return peaks of the mask's distance transform as circles (x, y, radius).

    Unlike Hough, results barely change between frames for stationary objects.
    Touching circles also split into separate peaks per center, so they can be separated.
    """
    if mask.size == 0:
        return []

    distance = _padded_distance(mask)

    window = max(3, int(min_radius) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window))
    peaks = (distance >= cv2.dilate(distance, kernel) - 1e-3) & (distance >= min_radius)

    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return []

    radii = distance[ys, xs]
    order = np.argsort(-radii)[:MAX_PEAK_CANDIDATES]

    circles: list[tuple[float, float, float]] = []
    for index in order:
        radius = float(radii[index])
        if radius > max_radius:
            continue

        x = int(xs[index])
        y = int(ys[index])

        if not _is_apex(distance, x, y, radius):
            continue

        if any(
            np.hypot(x - cx, y - cy) < max(radius, cr) * NMS_RATIO
            for cx, cy, cr in circles
        ):
            continue

        circles.append((float(x), float(y), radius))

    return circles


def _is_apex(distance: np.ndarray, x: int, y: int, radius: float) -> bool:
    """Whether it is a maximum over an area proportionate to its own radius.

    Between touching fruits a ridge forms where the inscribed circle grows toward either center.
    Peaks rise on ridges too, but there is no ball there. For a real one
    the inscribed circle always shrinks when moved slightly, so a summit test tells them apart.

    The window that picks up peaks is a fixed width matched to the smallest fruit, and does not reach wide ridges
    between big fruits. Re-check it scaled to the radius.
    """
    reach = max(1, int(radius * APEX_REACH_RATIO))
    height, width = distance.shape

    top, bottom = max(0, y - reach), min(height, y + reach + 1)
    left, right = max(0, x - reach), min(width, x + reach + 1)

    rows = np.arange(top, bottom)[:, None]
    columns = np.arange(left, right)[None, :]
    inside = (columns - x) ** 2 + (rows - y) ** 2 <= reach * reach

    return bool(
        distance[top:bottom, left:right][inside].max() <= distance[y, x] * APEX_TOLERANCE
    )


def _padded_distance(mask: np.ndarray) -> np.ndarray:
    """Pad with 0 before the distance transform so regions touching the image edge do not get oversized radii."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    return distance[1:-1, 1:-1]
