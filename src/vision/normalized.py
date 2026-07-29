"""The coordinate system of the board seen from the front.

Both board fruits and the waiting fruit have positions and radii in this coordinate system.
Inside the board y is 0-NORMALIZED_HEIGHT, and above the top edge y is negative.
"""

import cv2
import numpy as np

NORMALIZED_WIDTH = 400
NORMALIZED_HEIGHT = 500

# The four corners of the board seen from the front. The destination of warp and the starting point of the inverse transform.
NORMALIZED_CORNERS = np.array(
    [
        [0, 0],
        [NORMALIZED_WIDTH - 1, 0],
        [NORMALIZED_WIDTH - 1, NORMALIZED_HEIGHT - 1],
        [0, NORMALIZED_HEIGHT - 1],
    ],
    dtype=np.float32,
)


def warp_matrix(corners: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(corners.astype(np.float32), NORMALIZED_CORNERS)


def inverse_warp_matrix(corners: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(NORMALIZED_CORNERS, corners.astype(np.float32))


def transform_point(matrix: np.ndarray, x: float, y: float) -> tuple[int, int]:
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, matrix)[0, 0]

    return int(round(transformed[0])), int(round(transformed[1]))
