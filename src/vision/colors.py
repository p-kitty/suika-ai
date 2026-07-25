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

# VRC版: 半径比が分類の主軸 (screenshot 実測ベース)
FRUIT_RADIUS_RATIO = [
    (0.030, 0.044),   # cherry
    (0.052, 0.072),   # strawberry
    (0.068, 0.086),   # grape
    (0.086, 0.104),   # dekopon
    (0.100, 0.120),   # orange
    (0.125, 0.160),   # apple
    (0.108, 0.128),   # pear
    (0.128, 0.152),   # peach
    (0.142, 0.168),   # pineapple
    (0.158, 0.182),   # melon
    (0.175, 0.215),   # watermelon
]

BOARD_BG_HSV = ((10, 15, 130), (35, 100, 245))

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
