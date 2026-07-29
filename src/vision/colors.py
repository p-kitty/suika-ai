import cv2
import numpy as np

from ..config import load

# 新しく出てくるフルーツの最大段階 (orange)。apple 以上は来ない。
# 落下待ちのものと next の泡のどちらにも当てはまる。
SPAWN_MAX_TYPE = 4

FRUIT_NAMES = [
    "cherry",
    "strawberry",
    "grape",
    "dekopon",
    "orange",
    "apple",
    "pear",
    "peach",
    "pineapple",
    "melon",
    "watermelon",
]

# 段階ごとの半径比 (watermelon = 1.0)。スイカゲームの当たり判定の半径
# (16.5 〜 129.5) をそのまま割ったもので、この比は skin が変わっても共通。
# 等比ではなく、grape/dekopon や apple/pear のように 1.13 倍しか違わない
# 隣接段階がある。そこは半径では割り切れないので色で決まる。
FRUIT_RELATIVE_RADIUS = [
    0.127,   # cherry
    0.185,   # strawberry
    0.236,   # grape
    0.266,   # dekopon
    0.344,   # orange
    0.440,   # apple
    0.498,   # pear
    0.602,   # peach
    0.683,   # pineapple
    0.849,   # melon
    1.000,   # watermelon
]

# watermelon の半径 / 盤面幅。実測に合わせて config の
# watermelon_radius_ratio で上書きする。
DEFAULT_WATERMELON_RATIO = 0.24

# 盤面の下地 (ベージュ)。影で暗くなっても外せるよう V は絞らない。
BOARD_BG_HSV = ((10, 0, 55), (35, 100, 255))

# 盤面の枠線。彩度が高くフルーツと色で区別できないため、
# 境界付近に限定して除去する。
# 盤面は上から下へ暖色に転ぶグラデーションで、下端は H=16 まで落ちる。
# 下限を締めると床際の帯がマスクから抜け、盤面の下辺を取り逃がす。
BOARD_FRAME_HSV = ((14, 60, 100), (45, 255, 255))

# フルーツはどれも鮮やかで、下地との差は彩度に出る。
DEFAULT_FRUIT_SATURATION_MIN = 95

COLOR_FAMILIES = {
    "red_orange": [0, 1, 3, 4, 5],
    "purple": [2],
    "yellow_green": [6, 8, 9, 10],
    "pink": [7],
}


def saturated_mask(hsv: np.ndarray) -> np.ndarray:
    """彩度でフルーツらしい画素だけを残す。"""
    saturation_min = load().get("fruit_saturation_min", DEFAULT_FRUIT_SATURATION_MIN)
    return cv2.inRange(hsv, (0, saturation_min, 45), (180, 255, 255))


def color_family(h: float, s: float) -> str:
    if s < 70:
        return "unknown"
    if 125 <= h <= 155:
        return "purple"
    if 148 <= h <= 172:
        return "pink"
    # VRC: 赤系フルーツはすべて H<=22 に潰れる。pear と pineapple は
    # H=27-28 とわずかに上で、そこが赤系と黄緑系の境になる。
    if h <= 24 or h >= 165:
        return "red_orange"
    if h <= 95:
        return "yellow_green"
    return "unknown"
