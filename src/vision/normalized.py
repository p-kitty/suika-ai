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


def warp_window(
    frame: np.ndarray,
    corners: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
) -> np.ndarray:
    """この座標系で指定した窓を、画面から起こす。

    盤面を起こすのと同じ射影に、窓の左上を原点に持ってくる平行移動を足す。
    盤面の外 (上辺より上、右横) を見るのに使う。盤面の外には合わせる目印が
    ないので射影の外挿になるが、盤面の面の遠近をそのまま引き継ぐので、
    画面の中のどこに写っていても同じ尺度で測れる。
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
    """この座標系の円を、画面の上の中心と半径に直す。"""
    center = transform_point(matrix, x, y)
    edge = transform_point(matrix, x + radius, y)

    return center, max(2, int(np.hypot(edge[0] - center[0], edge[1] - center[1])))
