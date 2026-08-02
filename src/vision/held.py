from dataclasses import dataclass

import cv2
import numpy as np

from ..draw import Color, put_text
from .blobs import circle_peaks, solid_mask
from .classify import ClassifyResult, classify, fruit_radius_ratios, sample_hsv
from .colors import SPAWN_MAX_TYPE, vivid_mask
from .normalized import (
    NORMALIZED_WIDTH,
    inverse_warp_matrix,
    screen_circle,
    warp_window,
)

# Height of the band looked at above the board. Tall enough for the waiting fruit and the cloud holding it.
BAND_HEIGHT = 140

# The drop point sits at a fixed height in the world, so how far above the board's top edge it is
# does not change when the view moves. 10 measurements fall within 57-66.
DROP_HEIGHT = 61.0
# Fruits stacked past the rim measure 39 or less. Taken wide enough not to reach there.
DROP_HEIGHT_TOLERANCE = 15.0

# The waiting fruit appears slightly smaller than fruits on the board. It lies where the projection is extended
# beyond the top edge, so its scale differs slightly from inside the board.
HELD_RADIUS_SCALE = 0.93


@dataclass
class HeldResult:
    """The next fruit to fall, held by the cloud."""

    fruit: ClassifyResult | None
    # Coordinates on the normalized board. Above the top edge, so y is negative. x is the drop column as is.
    x: float | None = None
    y: float | None = None
    radius: float | None = None
    radius_ratio: float | None = None


def detect(frame: np.ndarray, corners: np.ndarray) -> HeldResult:
    band = _warp_band(frame, corners)
    mask = _band_mask(band)
    blob = _find_blob(mask)
    if blob is None:
        return HeldResult(fruit=None)

    x, y, radius = blob
    radius_ratio = radius / (NORMALIZED_WIDTH * HELD_RADIUS_SCALE)

    hsv_mean = sample_hsv(band, x, y, radius, valid_mask=mask)
    fruit = classify(radius_ratio, hsv_mean, max_type=SPAWN_MAX_TYPE)

    return HeldResult(
        fruit=fruit,
        x=x,
        y=y - BAND_HEIGHT,
        radius=radius,
        radius_ratio=radius_ratio,
    )


def draw_debug(frame: np.ndarray, corners: np.ndarray, result: HeldResult) -> None:
    label, color = _label(result)

    if result.x is None or result.y is None or result.radius is None:
        put_text(frame, label, (8, 104), color)
        return

    center, radius = screen_circle(
        inverse_warp_matrix(corners), result.x, result.y, result.radius
    )

    cv2.circle(frame, center, radius, color, 2)
    cv2.circle(frame, center, 2, color, -1)

    origin = (center[0] - radius, max(12, center[1] - radius - 6))
    put_text(frame, label, origin, color, scale=0.45, thickness=1)


def _label(result: HeldResult) -> tuple[str, Color]:
    if result.fruit is not None:
        return f"held: {result.fruit.name} {result.fruit.confidence:.0f}%", (0, 255, 255)
    if result.radius_ratio is not None:
        return f"held: --- r={result.radius_ratio:.3f}", (0, 165, 255)
    return "held: ---", (0, 0, 255)


def _warp_band(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the area just above the board's top edge. x within the band is the drop column as is."""
    return warp_window(frame, corners, 0, -BAND_HEIGHT, NORMALIZED_WIDTH, BAND_HEIGHT)


def _band_mask(band: np.ndarray) -> np.ndarray:
    """Only fruits remain in the band. The night sky behind is dark and the holding cloud is pale."""
    return solid_mask(vivid_mask(cv2.cvtColor(band, cv2.COLOR_BGR2HSV)))


def _find_blob(mask: np.ndarray) -> tuple[float, float, float] | None:
    """Return the blob at the height of the drop point.

    The band also shows fruits on their way down and fruits stacked on the board past the rim.
    Only the waiting one sits at the drop point's height, so it is chosen by the offset from there.
    """
    # Only cherry-orange can be dropped. Sizes outside that range are looking at something else.
    ratios = fruit_radius_ratios()
    min_radius = max(2.0, NORMALIZED_WIDTH * HELD_RADIUS_SCALE * ratios[0] * 0.6)
    max_radius = max(
        min_radius + 1.0,
        NORMALIZED_WIDTH * HELD_RADIUS_SCALE * ratios[SPAWN_MAX_TYPE] * 1.4,
    )

    mask = _without_overhang(mask)

    candidates = [
        peak
        for peak in circle_peaks(mask, min_radius, max_radius)
        if _height_error(peak[1]) <= DROP_HEIGHT_TOLERANCE
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda peak: _height_error(peak[1]))


def _without_overhang(mask: np.ndarray) -> np.ndarray:
    """Remove fruits past the rim from the band mask.

    Overflowing fruits touch the bottom of the band. They may be connected to held, so
    first cut a thin strip at the bottom to separate them, then drop the remaining blobs 'toward the bottom'.
    """
    if mask.size == 0 or not np.any(mask):
        return mask

    height = mask.shape[0]
    strip = max(4, height // 16)
    severed = mask.copy()
    severed[-strip:, :] = 0

    num, labels, _stats, centroids = cv2.connectedComponentsWithStats(
        (severed > 0).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return severed

    # A blob whose centroid is well below the drop point is an intrusion from the rim.
    limit = BAND_HEIGHT - DROP_HEIGHT + DROP_HEIGHT_TOLERANCE
    clear = np.zeros(num, dtype=bool)
    for label in range(1, num):
        if centroids[label][1] > limit:
            clear[label] = True

    if not np.any(clear):
        return severed

    label_ids = np.asarray(labels, dtype=np.intp)
    return np.where(clear[label_ids], 0, severed).astype(severed.dtype)


def _height_error(y: float) -> float:
    return abs((BAND_HEIGHT - y) - DROP_HEIGHT)
