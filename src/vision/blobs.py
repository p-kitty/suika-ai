import cv2
import numpy as np

MAX_PEAK_CANDIDATES = 400
# ピーク同士がこの割合より近ければ同じ円とみなす。
NMS_RATIO = 0.65


def circle_peaks(
    mask: np.ndarray,
    min_radius: float,
    max_radius: float,
) -> list[tuple[float, float, float]]:
    """マスクの距離変換のピークを円 (x, y, radius) として返す。

    Hough と違って静止した対象ならフレーム間で結果がほぼ変わらない。
    接触した円も中心ごとにピークが分かれるので分離できる。
    """
    if mask.size == 0:
        return []

    distance = _padded_distance(mask)

    window = max(3, int(min_radius) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window))
    peaks = (distance >= cv2.dilate(distance, kernel) - 1e-3) & (distance >= min_radius)

    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return []

    radii = distance[ys, xs]
    order = np.argsort(-radii)[:MAX_PEAK_CANDIDATES]

    circles: list[tuple[float, float, float]] = []
    for index in order:
        radius = float(radii[index])
        if radius > max_radius:
            continue

        x = float(xs[index])
        y = float(ys[index])

        if any(
            np.hypot(x - cx, y - cy) < max(radius, cr) * NMS_RATIO
            for cx, cy, cr in circles
        ):
            continue

        circles.append((x, y, radius))

    return circles


def _padded_distance(mask: np.ndarray) -> np.ndarray:
    """画像端に接した領域の半径が過大にならないよう 0 で囲んでから距離変換する。"""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    return distance[1:-1, 1:-1]
