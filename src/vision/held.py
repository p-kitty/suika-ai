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

# 盤面の上に見る帯の高さ。落下待ちフルーツと、それを持つ雲が収まる高さ。
BAND_HEIGHT = 140

# 落下点は世界の中で決まった高さにあるので、盤面の上辺からどれだけ上かは
# 視点が動いても変わらない。10 枚の実測は 57〜66 に収まる。
DROP_HEIGHT = 61.0
# 縁を越えて積み上がったフルーツは実測で 39 以下。そこへ届かない広さに取る。
DROP_HEIGHT_TOLERANCE = 15.0

# 帯の背景は鮮やかだが暗く (V=15〜70)、持ち手の雲は明るいが淡い (S=105)。
# 明るくて鮮やかなのはフルーツだけなので、両方で切れば残るのはフルーツ。
DEFAULT_SATURATION_MIN = 130
DEFAULT_VALUE_MIN = 110

# 落下待ちフルーツは盤面のフルーツより少し小さく写る。上辺の外へ射影を
# 伸ばした先なので、盤面の中と尺度がわずかに違う。
DEFAULT_RADIUS_SCALE = 0.93


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
    """盤面の上辺より上を、盤面と同じ向き・同じ尺度に起こす。

    盤面を起こすのと同じ射影に、帯のぶんだけ下へずらす平行移動を足す。
    こうすると帯の中の x はそのまま落とす列になり、半径も盤面のフルーツと
    同じ尺度で読める。
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
    """囲まれた穴を埋める。

    フルーツの顔の模様は暗く、照り返しは淡いのでマスクから抜け、内側に穴が
    残る。穴があると距離変換のピークが中心から追い出され、一つのフルーツが
    小さな円いくつかに割れる。帯に残るのはフルーツだけなので、囲まれた穴は
    すべてフルーツの内側のものとして埋めてよい。
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
    """落下点の高さにいる塊を返す。

    帯には落ちていく途中のフルーツや、盤面に積み上がって縁を越えたフルーツも
    写る。落下待ちのものだけが落下点の高さにいるので、そこからのずれで選ぶ。
    """
    scale = _radius_scale()

    # 落とせるのは cherry〜orange だけ。その範囲外の大きさは別のものを見ている。
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
