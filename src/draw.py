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
