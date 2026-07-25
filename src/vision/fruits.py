import cv2
import numpy as np

from ..config import load
from .blobs import circle_peaks
from .classify import classify, fruit_radius_ratios, sample_hsv
from .colors import BOARD_BG_HSV, DEFAULT_FRUIT_SATURATION_MIN
from .state import Fruit

# 検出した四隅は枠の外側なので、warp した盤面には枠と内側の影が写る。
# その帯を落とすと、壁に接したフルーツの境界が正しく内壁になる。
BORDER_BAND_RATIO = 0.045

# 下地の彩度はフルーツと重なるため、閾値だけでは分離できない。
# 円周にどれだけ輪郭が乗っているかで「実際に見えている球」だけを残す。
EDGE_SUPPORT_SAMPLES = 48
# 重なったフルーツは円周の一部が隠れるので、低めに取る。
MIN_EDGE_SUPPORT = 0.25
EDGE_GRADIENT_THRESHOLD = 60.0


def detect(board: np.ndarray) -> list[Fruit]:
    if board.size == 0:
        return []

    mask = _fruit_mask(board)
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


def _fruit_mask(board: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    saturation_min = load().get("fruit_saturation_min", DEFAULT_FRUIT_SATURATION_MIN)

    fruit_mask = cv2.inRange(hsv, (0, saturation_min, 45), (180, 255, 255))

    bg_lower, bg_upper = BOARD_BG_HSV
    fruit_mask[cv2.inRange(hsv, bg_lower, bg_upper) > 0] = 0

    # 枠・影・枠の外の背景はいずれも彩度が高く色では切れないので、
    # 盤面の縁は色を見ずにまとめて落とす。
    fruit_mask[_border_band(fruit_mask.shape)] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)

    return fruit_mask


def _border_band(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    band = np.zeros(shape, dtype=bool)

    margin_y = max(1, int(height * BORDER_BAND_RATIO))
    margin_x = max(1, int(width * BORDER_BAND_RATIO))

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
            overlap = fruit.radius + existing.radius - distance
            if overlap > min(fruit.radius, existing.radius) * 0.50:
                duplicate = True
                break

        if not duplicate:
            kept.append(fruit)

    return kept
