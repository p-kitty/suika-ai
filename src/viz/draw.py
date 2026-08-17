import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

Color = tuple[int, int, int]
Point = tuple[int, int]

# 段階ごとの BGR。screenshots の盤面から中央付近をサンプリングした色。
# view_sim / preview のミニ盤で共有する。
FRUIT_BGR: list[Color] = [
    (2, 5, 199),       # cherry — 濃い赤
    (59, 90, 208),     # strawberry — 赤みの強い橙赤
    (212, 88, 134),    # grape — 紫
    (5, 155, 211),     # dekopon — 明るい橙
    (19, 113, 216),    # orange — やや赤みの橙 (柿色)
    (17, 15, 204),     # apple — 濃い赤
    (104, 202, 211),   # pear — 淡黄緑
    (145, 153, 213),   # peach — ピンク
    (4, 196, 206),     # pineapple — 黄
    (12, 185, 130),    # melon — 黄緑
    (6, 127, 14),      # watermelon — 濃い緑
]


def put_text(
    image: np.ndarray,
    text: str,
    origin: Point,
    color: Color,
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def mode_badge(image: np.ndarray, auto: bool, fast: bool = False) -> None:
    """画面右上に AUTO / LIVE (+FAST) を常時表示する。左上の board/next と被らない。"""
    label = "AUTO" if auto else "LIVE"
    if fast:
        label += "+FAST"
    fg: Color = (40, 255, 120) if auto else (170, 170, 170)
    bg: Color = (20, 70, 30) if auto else (36, 36, 36)
    border: Color = (60, 255, 160) if auto else (90, 90, 90)
    if fast:
        fg, bg, border = (60, 220, 255), (20, 50, 70), (80, 200, 255)
    scale = 1.15
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(label, FONT, scale, thickness)
    pad_x, pad_y = 12, 8
    x = image.shape[1] - tw - pad_x - 8
    y = 14 + th
    top_left = (x - pad_x, y - th - pad_y)
    bottom_right = (x + tw + pad_x, y + baseline + pad_y // 2)
    cv2.rectangle(image, top_left, bottom_right, bg, -1)
    cv2.rectangle(image, top_left, bottom_right, border, 2)
    put_text(image, label, (x, y), fg, scale=scale, thickness=thickness)
