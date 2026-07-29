import cv2
import numpy as np

MAX_PEAK_CANDIDATES = 400
# ピーク同士がこの割合より近ければ同じ円とみなす。
NMS_RATIO = 0.65

# ピークが頂点かどうかを見る範囲。自分の半径に対する割合。
APEX_REACH_RATIO = 0.7
# 距離変換は近似なので、平らな頂上でも高さがわずかにばらつく。実測では本物が
# 1.001 倍までに収まり、尾根の上のピークは 1.13 倍以上で、間は空いている。
APEX_TOLERANCE = 1.05


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

        x = int(xs[index])
        y = int(ys[index])

        if not _is_apex(distance, x, y, radius):
            continue

        if any(
            np.hypot(x - cx, y - cy) < max(radius, cr) * NMS_RATIO
            for cx, cy, cr in circles
        ):
            continue

        circles.append((float(x), float(y), radius))

    return circles


def solid_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """しきい値のざらつきを落とし、囲まれた穴を埋める。

    円を探す前の下準備。ざらつきは距離変換を削り、穴はピークを中心から
    追い出して一つの塊を小さな円いくつかに割ってしまう。

    穴を無条件に埋めてよいのは、塊の中にフルーツ以外が写らない場合だけ。
    盤面の中は触れ合ったフルーツに囲まれた下地も穴になるので、fruits 側は
    色を見て選んでいる。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return _fill_holes(mask)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    background = (mask == 0).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(background, connectivity=8)

    enclosed = np.ones(count, dtype=bool)
    enclosed[0] = False
    enclosed[_border_labels(labels)] = False

    filled = mask.copy()
    filled[enclosed[labels]] = 255

    return filled


def _border_labels(labels: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))


def _is_apex(distance: np.ndarray, x: int, y: int, radius: float) -> bool:
    """自分の半径に見合った広さで最大かどうか。

    触れ合ったフルーツの間には、どちらの中心へ寄っても内接円が大きくなる
    尾根ができる。尾根の上にもピークは立つが、そこに球はない。本物なら
    内接円は少し動かすと必ず小さくなるので、頂点かどうかで見分けられる。

    ピークを拾う窓は最小のフルーツに合わせた固定幅で、大きなフルーツの
    間にできる幅の広い尾根には届かない。半径に比例させて見直す。
    """
    reach = max(1, int(radius * APEX_REACH_RATIO))
    height, width = distance.shape

    top, bottom = max(0, y - reach), min(height, y + reach + 1)
    left, right = max(0, x - reach), min(width, x + reach + 1)

    rows = np.arange(top, bottom)[:, None]
    columns = np.arange(left, right)[None, :]
    inside = (columns - x) ** 2 + (rows - y) ** 2 <= reach * reach

    return bool(
        distance[top:bottom, left:right][inside].max() <= distance[y, x] * APEX_TOLERANCE
    )


def _padded_distance(mask: np.ndarray) -> np.ndarray:
    """画像端に接した領域の半径が過大にならないよう 0 で囲んでから距離変換する。"""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    return distance[1:-1, 1:-1]
