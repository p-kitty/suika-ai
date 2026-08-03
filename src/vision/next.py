import math
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

# 泡は盤面の右上あたりに浮いている。盤面の面に投げた影として見たときの中心で、
# 盤面が壁の内側基準になった後、11 枚すべて (605〜622, -32〜-18) に収まる。
BUBBLE_X = 619
BUBBLE_Y = -23

# 中心の周りに取る窓の半径。下の Merge Order の輪や右の木を入れない広さ。
WINDOW_HALF = 120
# 泡の中心からこれだけ離れた塊は next のフルーツではない。
CENTER_TOLERANCE = 54


@dataclass
class NextResult:
    """next の泡の中身。落下待ちのさらに次に来るフルーツ。"""

    fruit: ClassifyResult | None
    # 正規化した盤面の座標。盤面の右外なので x は幅より大きい。
    x: float | None = None
    y: float | None = None
    radius: float | None = None
    radius_ratio: float | None = None


def detect(frame: np.ndarray, corners: np.ndarray) -> NextResult:
    window = _warp_window(frame, corners)
    mask = _window_mask(window)

    blob = _find_blob(mask)
    if blob is None:
        return NextResult(fruit=None)

    x, y, radius = blob
    # 泡の中のフルーツは、盤面に置いたときとほぼ同じ大きさに写る
    # (実測 11 枚の中央値で 1.00 倍)。尺度の読み替えはいらない。
    radius_ratio = radius / NORMALIZED_WIDTH

    hsv_mean = sample_hsv(window, x, y, radius, valid_mask=mask)
    fruit = classify(radius_ratio, hsv_mean, max_type=SPAWN_MAX_TYPE)

    return NextResult(
        fruit=fruit,
        x=BUBBLE_X - WINDOW_HALF + x,
        y=BUBBLE_Y - WINDOW_HALF + y,
        radius=radius,
        radius_ratio=radius_ratio,
    )


def draw_debug(frame: np.ndarray, corners: np.ndarray, result: NextResult) -> None:
    label, color = _label(result)

    if result.x is None or result.y is None or result.radius is None:
        put_text(frame, label, (8, 52), color)
        return

    center, radius = screen_circle(
        inverse_warp_matrix(corners), result.x, result.y, result.radius
    )

    cv2.circle(frame, center, radius, color, 2)
    cv2.circle(frame, center, 2, color, -1)

    origin = (center[0] - radius, max(12, center[1] - radius - 6))
    put_text(frame, label, origin, color, scale=0.45, thickness=1)


def _label(result: NextResult) -> tuple[str, Color]:
    if result.fruit is not None:
        return f"next: {result.fruit.name} {result.fruit.confidence:.0f}%", (255, 0, 255)
    if result.radius_ratio is not None:
        return f"next: --- r={result.radius_ratio:.3f}", (0, 165, 255)
    return "next: ---", (0, 0, 255)


def _warp_window(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """泡のあたりを盤面と同じ尺度に起こす。

    盤面幅で割って測ると、視点を振ったときに狂う。盤面は画面の中央寄り、
    泡は端寄りに写るので、射影で引き伸ばされる量が両者で違ってくる。
    盤面と同じ射影で起こせば、泡の位置での引き伸ばしも一緒に戻る。
    """
    return warp_window(
        frame,
        corners,
        BUBBLE_X - WINDOW_HALF,
        BUBBLE_Y - WINDOW_HALF,
        WINDOW_HALF * 2,
        WINDOW_HALF * 2,
    )


def _window_mask(window: np.ndarray) -> np.ndarray:
    """泡の中のフルーツだけを残す。

    泡を包む玉と星は淡いので彩度で落ちるが、玉を透かして見える夜空は
    暗いだけで彩度は高い。彩度だけで切ると泡がまるごと残り、測るのは
    中のフルーツではなく泡の大きさになってしまう。
    """
    return solid_mask(vivid_mask(cv2.cvtColor(window, cv2.COLOR_BGR2HSV)))


def _find_blob(mask: np.ndarray) -> tuple[float, float, float] | None:
    # next は cherry〜orange しか出ないので、その範囲外の大きさは
    # 玉や背景を拾っているだけと判断できる。
    ratios = fruit_radius_ratios()
    min_radius = max(2.0, NORMALIZED_WIDTH * ratios[0] * 0.6)
    max_radius = max(min_radius + 1.0, NORMALIZED_WIDTH * ratios[SPAWN_MAX_TYPE] * 1.4)

    centered = [
        peak
        for peak in circle_peaks(mask, min_radius, max_radius)
        if math.hypot(peak[0] - WINDOW_HALF, peak[1] - WINDOW_HALF) <= CENTER_TOLERANCE
    ]
    if not centered:
        return None

    return max(centered, key=lambda peak: peak[2])
