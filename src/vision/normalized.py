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


def warp_window(
    frame: np.ndarray,
    corners: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Warp a window specified in this coordinate system from the screen.

    The same projection used to warp the board, plus a translation bringing the window's top left to the origin.
    Used to look outside the board (above the top edge, to the right). Outside the board there are no landmarks to match,
    so the projection is extrapolated, but it carries over the perspective of the board's plane as is,
    so wherever it appears on the screen it is measured at the same scale.
    """
    shift = np.array(
        [[1.0, 0.0, -float(left)], [0.0, 1.0, -float(top)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    return cv2.warpPerspective(frame, shift @ warp_matrix(corners), (width, height))


def screen_circle(
    matrix: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> tuple[tuple[int, int], int]:
    """Convert a circle in this coordinate system to a center and radius on the screen."""
    center = transform_point(matrix, x, y)
    edge = transform_point(matrix, x + radius, y)

    return center, max(2, int(np.hypot(edge[0] - center[0], edge[1] - center[1])))
