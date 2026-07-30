from functools import cache

import cv2
import numpy as np

from .blobs import border_labels, circle_peaks
from .classify import classify, fruit_radius_ratios, sample_hsv
from .colors import BOARD_BG_HSV, saturated_mask
from .state import Fruit

# 検出した四隅は枠の外側なので、warp した盤面には枠と内側の影が写る。
# その帯を落とすと、壁に接したフルーツの境界が正しく内壁になる。
BORDER_BAND_RATIO = 0.045

# 下地は上から下へ滑らかに濃くなるグラデーションで、下側はフルーツと同じ
# くらい彩度が高い。固定しきい値では切れないので、下地の色を座標の一次式
# として当てはめ、そこからの色差でフルーツを取る。
BACKGROUND_TOLERANCE = 14.0
BACKGROUND_FIT_ITERATIONS = 5
# 平面を決めるのに全画素はいらない。間引いて当てはめる。
BACKGROUND_FIT_STRIDE = 4
# 間引いたあとの画素数。種がフルーツで埋まったら当てはめは失敗とみなす。
MIN_BACKGROUND_SAMPLES = 300
# 種を取るリングの幅。縁の帯のすぐ内側から取る。
BACKGROUND_SEED_WIDTH_RATIO = 0.03

# 下地の彩度はフルーツと重なるため、閾値だけでは分離できない。
# 円周にどれだけ輪郭が乗っているかで「実際に見えている球」だけを残す。
EDGE_SUPPORT_SAMPLES = 48
# 重なったフルーツは円周の一部が隠れるので、低めに取る。
MIN_EDGE_SUPPORT = 0.25
EDGE_GRADIENT_THRESHOLD = 60.0

# 囲まれた穴を照り返しと見なす、下地の色からの距離。実測では照り返しが
# 12〜20、触れ合ったフルーツに囲まれた本物の下地が 1〜4 で、間は空いている。
HOLE_BACKGROUND_DISTANCE = 8.0


def detect(board: np.ndarray) -> list[Fruit]:
    if board.size == 0:
        return []

    mask = fruit_mask(board)
    outline = _outline(board)
    fruits = []

    for x, y, radius in _find_circles(board.shape[1], mask):
        if _edge_support(outline, x, y, radius) < MIN_EDGE_SUPPORT:
            continue

        fruit = _classify(board, mask, x, y, radius)
        if fruit is not None:
            fruits.append(fruit)

    return _deduplicate(fruits)


def _outline(board: np.ndarray) -> np.ndarray:
    """画像上に実際に見えている輪郭。

    マスクの境界は使わない。しきい値が下地を拾ってできた塊は自分の境界を
    必ず持つので、それを輪郭に含めると本物の球と区別できなくなる。

    明度だけを見ると、下地と明るさが近く色だけが違うフルーツ (ベージュ上の
    洋なしなど) の輪郭を取り逃がす。Lab の全チャンネルで勾配を取る。
    """
    lab = cv2.GaussianBlur(cv2.cvtColor(board, cv2.COLOR_BGR2LAB), (5, 5), 0)

    gradient = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel in range(3):
        plane = lab[:, :, channel]
        gx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.maximum(gradient, cv2.magnitude(gx, gy))

    edges = (gradient >= EDGE_GRADIENT_THRESHOLD).astype(np.uint8) * 255

    return cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def _edge_support(outline: np.ndarray, x: float, y: float, radius: float) -> float:
    height, width = outline.shape[:2]
    angles = np.linspace(0.0, 2.0 * np.pi, EDGE_SUPPORT_SAMPLES, endpoint=False)

    xs = np.clip((x + radius * np.cos(angles)).astype(int), 0, width - 1)
    ys = np.clip((y + radius * np.sin(angles)).astype(int), 0, height - 1)

    return float((outline[ys, xs] > 0).mean())


def fruit_mask(board: np.ndarray) -> np.ndarray:
    distance = _background_distance(board)

    if distance is None:
        mask = _saturation_mask(board)
    else:
        mask = (distance > BACKGROUND_TOLERANCE).astype(np.uint8) * 255

    # 枠・影・枠の外の背景はいずれも彩度が高く色では切れないので、
    # 盤面の縁は色を見ずにまとめて落とす。
    mask[_border_band(mask.shape)] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if distance is None:
        return mask

    return _fill_highlights(mask, distance)


def _fill_highlights(mask: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """フルーツの表面の照り返しが空けた穴を埋める。

    強い照り返しは下地のベージュに近い色に写るので許容範囲に収まってしまい、
    フルーツの内側に穴が残る。穴があると距離変換のピークが中心から追い出さ
    れ、一つのフルーツが小さな円いくつかに割れる。

    触れ合ったフルーツに囲まれた下地も同じく囲まれた穴になるが、そちらを
    埋めるとフルーツ同士がつながって巨大な円になる。見分けるのは下地の色から
    の距離で、照り返しは許容範囲の際にいて、本物の下地はずっと内側にいる。
    """
    background = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)

    # ラベル 0 はフルーツ自身。盤面の外まで続く領域は下地なので触らない。
    enclosed = np.ones(count, dtype=bool)
    enclosed[0] = False
    enclosed[border_labels(labels)] = False

    highlight = np.zeros(count, dtype=bool)
    for label in np.nonzero(enclosed)[0]:
        left, top, width, height = stats[label, :4]
        window = (slice(top, top + height), slice(left, left + width))
        hole = labels[window] == label

        highlight[label] = np.median(distance[window][hole]) >= HOLE_BACKGROUND_DISTANCE

    filled = mask.copy()
    filled[highlight[labels]] = 255

    return filled


def _background_distance(board: np.ndarray) -> np.ndarray | None:
    """各画素が下地の色からどれだけ離れているかを返す。

    下地は上から下へ滑らかに濃くなるだけなので、色を座標の一次式
    Lab = a + b*x + c*y で表せる。フルーツは外れ値として繰り返し
    切り落とし、残った画素だけで当てはめる。

    塗り広げと違って盤面が埋まっていても壊れず、フルーツに囲まれて
    孤立した下地も下地として拾える。
    """
    # メディアンは段差を保つので、縁がなだらかにならず色差が鈍らない。
    lab = cv2.cvtColor(cv2.medianBlur(board, 5), cv2.COLOR_BGR2LAB).astype(np.float32)

    coefficients = _fit_background(lab, _seed_band(board.shape[:2]))
    if coefficients is None:
        return None

    # 縁の帯だけでは種が画面の外周に偏る。一度当てた結果で下地と判定
    # できた画素を使って当て直すと、盤面全体に散った種で決まる。
    inliers = _distance_to(lab, coefficients) < BACKGROUND_TOLERANCE
    refitted = _fit_background(lab, inliers)

    return _distance_to(lab, coefficients if refitted is None else refitted)


def _fit_background(lab: np.ndarray, seeds: np.ndarray) -> np.ndarray | None:
    """一次式の係数 (3x3) を、外れ値を切り落としながら求める。"""
    stride = BACKGROUND_FIT_STRIDE
    design = _coordinate_design(lab.shape[:2])
    samples = lab[::stride, ::stride].reshape(-1, 3)
    keep = seeds[::stride, ::stride].reshape(-1).copy()

    for _ in range(BACKGROUND_FIT_ITERATIONS):
        if int(keep.sum()) < MIN_BACKGROUND_SAMPLES:
            return None

        rows = design[keep]
        coefficients = np.linalg.lstsq(rows, samples[keep], rcond=None)[0]
        residual = np.linalg.norm(rows @ coefficients - samples[keep], axis=1)

        outliers = residual >= BACKGROUND_TOLERANCE
        if not outliers.any():
            break

        keep[np.nonzero(keep)[0][outliers]] = False

    return coefficients


def _distance_to(lab: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """一次式が表す下地の色から、各画素がどれだけ離れているか。"""
    height, width = lab.shape[:2]
    constant, per_x, per_y = coefficients

    model = constant + np.arange(width, dtype=np.float32)[None, :, None] * per_x
    model = model + np.arange(height, dtype=np.float32)[:, None, None] * per_y

    return np.linalg.norm(lab - model, axis=2)


@cache
def _coordinate_design(shape: tuple[int, int]) -> np.ndarray:
    """当てはめ用に間引いた座標 [1, x, y]。平面を決めるのに全画素はいらない。"""
    height, width = shape
    stride = BACKGROUND_FIT_STRIDE
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]

    return np.stack([np.ones_like(xs), xs, ys], axis=-1).astype(np.float32).reshape(-1, 3)


def _seed_band(shape: tuple[int, int]) -> np.ndarray:
    """縁の帯のすぐ内側のリング。壁際は下地が見えていることが多い。"""
    outer = _border_band(shape)
    inner = _border_band(shape, BORDER_BAND_RATIO + BACKGROUND_SEED_WIDTH_RATIO)

    return inner & ~outer


def _saturation_mask(board: np.ndarray) -> np.ndarray:
    """当てはめが失敗したときの予備。"""
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    mask = saturated_mask(hsv)

    bg_lower, bg_upper = BOARD_BG_HSV
    mask[cv2.inRange(hsv, bg_lower, bg_upper) > 0] = 0

    return mask


def _border_band(shape: tuple[int, int], ratio: float = BORDER_BAND_RATIO) -> np.ndarray:
    height, width = shape
    band = np.zeros(shape, dtype=bool)

    margin_y = max(1, int(height * ratio))
    margin_x = max(1, int(width * ratio))

    band[:margin_y, :] = True
    band[height - margin_y :, :] = True
    band[:, :margin_x] = True
    band[:, width - margin_x :] = True

    return band


def _find_circles(
    board_width: int,
    mask: np.ndarray,
) -> list[tuple[float, float, float]]:
    ratios = fruit_radius_ratios()
    min_radius = max(3.0, board_width * ratios[0] * 0.7)
    max_radius = max(min_radius + 2.0, board_width * ratios[-1] * 1.3)

    return circle_peaks(mask, min_radius, max_radius)


def _classify(
    board: np.ndarray,
    mask: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> Fruit | None:
    radius_ratio = radius / board.shape[1]

    hsv_mean = sample_hsv(board, x, y, radius, valid_mask=mask)
    result = classify(radius_ratio, hsv_mean)

    if result is None:
        return None

    return Fruit(
        type=result.type,
        x=x,
        y=y,
        radius=radius,
        confidence=result.confidence,
    )


def _deduplicate(fruits: list[Fruit]) -> list[Fruit]:
    if not fruits:
        return []

    fruits = sorted(fruits, key=lambda fruit: (fruit.radius, fruit.confidence), reverse=True)
    kept = []

    for fruit in fruits:
        duplicate = False

        for existing in kept:
            distance = np.hypot(fruit.x - existing.x, fruit.y - existing.y)
            # 同じフルーツに立った二つ目のピークだけを落としたい。触れ合った
            # 別のフルーツは、見た目が多少重なっても中心までは食い込まない。
            if distance < max(fruit.radius, existing.radius):
                duplicate = True
                break

        if not duplicate:
            kept.append(fruit)

    return kept
