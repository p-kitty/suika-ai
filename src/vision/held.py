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

# 盤面の上に見る帯の高さ。落下待ちフルーツと、それを持つ雲が収まる高さ。
BAND_HEIGHT = 140

# 落下点は世界の中で決まった高さにあるので、盤面の上辺からどれだけ上かは
# 視点が動いても変わらない。10 枚の実測は 57〜66 に収まる。
DROP_HEIGHT = 61.0
# 縁を越えて積み上がったフルーツは実測で 39 以下。そこへ届かない広さに取る。
DROP_HEIGHT_TOLERANCE = 15.0

# 落下待ちフルーツは盤面のフルーツより少し小さく写る。上辺の外へ射影を
# 伸ばした先なので、盤面の中と尺度がわずかに違う。
HELD_RADIUS_SCALE = 0.93


@dataclass
class HeldResult:
    """雲が持っている、次に落ちるフルーツ。"""

    fruit: ClassifyResult | None
    # 正規化した盤面の座標。上辺より上なので y は負。x はそのまま落とす列。
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
    """盤面の上辺のすぐ上を起こす。帯の中の x はそのまま落とす列になる。"""
    return warp_window(frame, corners, 0, -BAND_HEIGHT, NORMALIZED_WIDTH, BAND_HEIGHT)


def _band_mask(band: np.ndarray) -> np.ndarray:
    """帯に残るのはフルーツだけ。背景の夜空は暗く、持ち手の雲は淡い。"""
    return solid_mask(vivid_mask(cv2.cvtColor(band, cv2.COLOR_BGR2HSV)))


def _find_blob(mask: np.ndarray) -> tuple[float, float, float] | None:
    """落下点の高さにいる塊を返す。

    帯には落ちていく途中のフルーツや、盤面に積み上がって縁を越えたフルーツも
    写る。落下待ちのものだけが落下点の高さにいるので、そこからのずれで選ぶ。
    """
    # 落とせるのは cherry〜orange だけ。その範囲外の大きさは別のものを見ている。
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
    """縁を越えた実を帯マスクから除く。

    はみ出し実は帯の下端に触れる。held とつながっていることもあるので、
    まず下端の細い帯を切って切り離し、残った「下寄り」の塊を落とす。
    """
    if mask.size == 0 or not np.any(mask):
        return mask

    height = mask.shape[0]
    strip = max(4, height // 16)
    severed = mask.copy()
    severed[-strip:, :] = 0

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (severed > 0).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return severed

    # 落下点より十分下に重心がある塊は縁からの侵入。
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
