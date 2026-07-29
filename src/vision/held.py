from dataclasses import dataclass

import cv2
import numpy as np

from ..config import load
from ..draw import Color, put_text
from .blobs import circle_peaks
from .classify import ClassifyResult, classify, fruit_radius_ratios, sample_hsv
from .colors import SPAWN_MAX_TYPE
from .normalized import (
    NORMALIZED_WIDTH,
    inverse_warp_matrix,
    transform_point,
    warp_matrix,
)

# Height of the band looked at above the board. Tall enough for the waiting fruit and the cloud holding it.
BAND_HEIGHT = 140

# The drop point sits at a fixed height in the world, so how far above the board's top edge it is
# does not change when the view moves. 10 measurements fall within 57-66.
DROP_HEIGHT = 61.0
# Fruits stacked past the rim measure 39 or less. Taken wide enough not to reach there.
DROP_HEIGHT_TOLERANCE = 15.0

# The band background is vivid but dark (V=15-70), and the holding cloud is bright but pale (S=105).
# Only fruits are bright and vivid, so cutting on both leaves only fruits.
DEFAULT_SATURATION_MIN = 130
DEFAULT_VALUE_MIN = 110

# The waiting fruit appears slightly smaller than fruits on the board. It lies where the projection is extended
# beyond the top edge, so its scale differs slightly from inside the board.
DEFAULT_RADIUS_SCALE = 0.93


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
    radius_ratio = radius / (NORMALIZED_WIDTH * _radius_scale())

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

    matrix = inverse_warp_matrix(corners)
    center = transform_point(matrix, result.x, result.y)
    edge = transform_point(matrix, result.x + result.radius, result.y)
    radius = max(2, int(np.hypot(edge[0] - center[0], edge[1] - center[1])))

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
    """Warp above the board's top edge in the same orientation and scale as the board.

    The same projection used to warp the board, plus a translation shifting down by the band.
    This makes x within the band the drop column as is, and radii readable at the same scale
    as the board's fruits.
    """
    shift = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, float(BAND_HEIGHT)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    return cv2.warpPerspective(
        frame,
        shift @ warp_matrix(corners),
        (NORMALIZED_WIDTH, BAND_HEIGHT),
    )


def _band_mask(band: np.ndarray) -> np.ndarray:
    cfg = load()
    saturation_min = cfg.get("held_saturation_min", DEFAULT_SATURATION_MIN)
    value_min = cfg.get("held_value_min", DEFAULT_VALUE_MIN)

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, saturation_min, value_min), (180, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return _fill_holes(mask)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes.

    The face patterns of fruits are dark and highlights are pale, so they drop out of the mask and leave holes
    inside. A hole pushes the distance transform's peak away from the center, and one fruit
    splits into several small circles. Only fruits remain in the band, so every enclosed hole
    can be filled as being inside a fruit.
    """
    background = (mask == 0).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(background, connectivity=8)

    enclosed = np.ones(count, dtype=bool)
    enclosed[0] = False
    enclosed[_border_labels(labels)] = False

    filled = mask.copy()
    filled[enclosed[labels]] = 255

    return filled


def _border_labels(labels: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))


def _find_blob(mask: np.ndarray) -> tuple[float, float, float] | None:
    """Return the blob at the height of the drop point.

    The band also shows fruits on their way down and fruits stacked on the board past the rim.
    Only the waiting one sits at the drop point's height, so it is chosen by the offset from there.
    """
    scale = _radius_scale()

    # Only cherry-orange can be dropped. Sizes outside that range are looking at something else.
    ratios = fruit_radius_ratios()
    min_radius = max(2.0, NORMALIZED_WIDTH * scale * ratios[0] * 0.6)
    max_radius = max(min_radius + 1.0, NORMALIZED_WIDTH * scale * ratios[SPAWN_MAX_TYPE] * 1.4)

    candidates = [
        peak
        for peak in circle_peaks(mask, min_radius, max_radius)
        if _height_error(peak[1]) <= DROP_HEIGHT_TOLERANCE
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda peak: _height_error(peak[1]))


def _height_error(y: float) -> float:
    return abs((BAND_HEIGHT - y) - DROP_HEIGHT)


def _radius_scale() -> float:
    return load().get("held_radius_scale", DEFAULT_RADIUS_SCALE) or DEFAULT_RADIUS_SCALE
