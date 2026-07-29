"""正面から見た盤面の座標系。

盤面のフルーツも落下待ちフルーツも、この座標系で位置と半径を持つ。
盤面の中は y が 0〜NORMALIZED_HEIGHT で、上辺より上は y が負になる。
"""

import cv2
import numpy as np

NORMALIZED_WIDTH = 400
NORMALIZED_HEIGHT = 500

# 正面から見た盤面の四隅。warp の行き先であり、逆変換の出発点でもある。
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
