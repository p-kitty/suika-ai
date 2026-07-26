# next に出現する最大段階 (orange)。apple 以上は来ない。
NEXT_MAX_TYPE = 4

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

# 段階ごとの半径比 (watermelon = 1.0)。スイカゲームの当たり判定は
# 段階が上がるごとに約 1.2 倍で、この比は skin が変わっても共通。
FRUIT_RELATIVE_RADIUS = [
    0.084,   # cherry
    0.130,   # strawberry
    0.175,   # grape
    0.234,   # dekopon
    0.299,   # orange
    0.383,   # apple
    0.481,   # pear
    0.591,   # peach
    0.721,   # pineapple
    0.838,   # melon
    1.000,   # watermelon
]

# watermelon の半径 / 盤面幅。実測に合わせて config の
# watermelon_radius_ratio で上書きする。
DEFAULT_WATERMELON_RATIO = 0.24

# 盤面の下地 (ベージュ)。影で暗くなっても外せるよう V は絞らない。
BOARD_BG_HSV = ((10, 0, 55), (35, 100, 255))

# 盤面の枠線。彩度が高くフルーツと色で区別できないため、
# 境界付近に限定して除去する。
BOARD_FRAME_HSV = ((18, 60, 100), (45, 255, 255))

# フルーツはどれも鮮やかで、下地との差は彩度に出る。
DEFAULT_FRUIT_SATURATION_MIN = 95

COLOR_FAMILIES = {
    "red_orange": [0, 1, 3, 4, 5],
    "purple": [2],
    "yellow_green": [6, 8, 9, 10],
    "pink": [7],
}


def color_family(h: float, s: float) -> str:
    if s < 70:
        return "unknown"
    if 125 <= h <= 155:
        return "purple"
    if 148 <= h <= 172:
        return "pink"
    if 30 <= h <= 95:
        return "yellow_green"
    # VRC: 赤系フルーツはすべて H≈15-25 に集まる
    if h <= 28 or h >= 165:
        return "red_orange"
    if 15 <= h <= 35:
        return "yellow_green"
    return "unknown"
