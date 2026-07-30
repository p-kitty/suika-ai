import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

Color = tuple[int, int, int]
Point = tuple[int, int]


def put_text(
    image: np.ndarray,
    text: str,
    origin: Point,
    color: Color,
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def mode_badge(image: np.ndarray, auto: bool) -> None:
    """画面左上に AUTO / LIVE を常時表示する。一時メッセージに消されない。"""
    label = "AUTO" if auto else "LIVE"
    fg: Color = (40, 255, 120) if auto else (170, 170, 170)
    bg: Color = (20, 70, 30) if auto else (36, 36, 36)
    border: Color = (60, 255, 160) if auto else (90, 90, 90)
    scale = 1.15
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(label, FONT, scale, thickness)
    x, y = 14, 42
    pad_x, pad_y = 12, 8
    top_left = (x - pad_x, y - th - pad_y)
    bottom_right = (x + tw + pad_x, y + baseline + pad_y // 2)
    cv2.rectangle(image, top_left, bottom_right, bg, -1)
    cv2.rectangle(image, top_left, bottom_right, border, 2)
    put_text(image, label, (x, y), fg, scale=scale, thickness=thickness)
