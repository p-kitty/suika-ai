from dataclasses import dataclass

import cv2
import numpy as np

from ..config import load
from ..draw import put_text
from .colors import BOARD_FRAME_HSV
from .dialog import is_blocked
from .fruits import detect as detect_fruits
from .held import HeldResult, detect as detect_held, draw_debug as draw_held_debug
from .next import NextResult, detect as detect_next, draw_debug as draw_next_debug
from .normalized import (
    NORMALIZED_HEIGHT,
    NORMALIZED_WIDTH,
    inverse_warp_matrix,
    screen_circle,
    warp_matrix,
)
from .state import Fruit

MIN_BOX_AREA_RATIO = 0.02

# 画面の端に接した四角形は盤面が画面外で切れている。角の位置を信用できず、
# warp すると枠の外まで写り込んで、そこに無いフルーツを拾ってしまう。
EDGE_MARGIN_RATIO = 0.01

CORNER_SMOOTH_ALPHA = 0.35
# これを超える移動は本当に視点が動いたと見なして平滑化せず追従する。
CORNER_JUMP_RATIO = 0.08

# 各辺に直線を当てるとき、角の丸みを避けて両端を捨てる割合。
CORNER_ARC_SKIP_RATIO = 0.15
# 直線を決めるのに要る点数。これを割る辺があれば当てはめを諦める。
MIN_SIDE_POINTS = 10
# 交点がこれだけ動いたら当てはめが破綻したとみなし、粗い四角形に戻す。
MAX_CORNER_SHIFT_RATIO = 0.25

# _find_corners が拾う四隅は箱の外側の装飾枠で、本当の透明な壁はそこから
# 内側にある。壁際で静止しているフルーツの実測 (screenshots/ 全数、壁に
# 接した検出済みフルーツと壁との隙間) から求めた比率。横と縦で枠の見え方が
# 違うため別々の値になる。
WALL_INSET_X_RATIO = 0.0825
WALL_INSET_Y_RATIO = 0.0534


@dataclass
class BoardResult:
    normalized: np.ndarray | None
    corners: np.ndarray | None
    found: bool
    # 次に落ちるフルーツ (雲が持っている) と、その次 (Next の泡)。
    held_fruit: HeldResult | None = None
    next_fruit: NextResult | None = None
    fruits: list[Fruit] | None = None
    # ダイアログが被っていて盤面を読めない状態。found とは別に持つ。
    blocked: bool = False


def localize(
    frame: np.ndarray,
    previous_corners: np.ndarray | None = None,
) -> BoardResult:
    corners = _find_corners(frame)
    if corners is None:
        return BoardResult(normalized=None, corners=None, found=False)

    if previous_corners is not None:
        corners = _smooth_corners(previous_corners, corners)

    normalized = _warp(frame, corners)

    if is_blocked(normalized):
        return BoardResult(
            normalized=normalized,
            corners=corners,
            found=True,
            blocked=True,
        )

    return BoardResult(
        normalized=normalized,
        corners=corners,
        found=True,
        held_fruit=detect_held(frame, corners),
        next_fruit=detect_next(frame, corners),
        fruits=detect_fruits(normalized),
    )


def draw_frame_debug(frame: np.ndarray, result: BoardResult) -> np.ndarray:
    output = frame.copy()

    if not (result.found and result.corners is not None):
        put_text(output, "board: not found", (8, 24), (0, 0, 255))
        return output

    _draw_corners(output, result.corners)

    if result.blocked:
        put_text(output, "board: dialog", (8, 24), (0, 165, 255))
        return output

    put_text(output, "board: detected", (8, 24), (0, 255, 0))

    if result.held_fruit is not None:
        draw_held_debug(output, result.corners, result.held_fruit)

    if result.next_fruit is not None:
        draw_next_debug(output, result.corners, result.next_fruit)

    if result.fruits is not None:
        _draw_fruits(output, result.corners, result.fruits)

    return output


def _draw_corners(output: np.ndarray, corners: np.ndarray) -> None:
    corners = corners.astype(int)
    cv2.polylines(output, [corners], True, (0, 255, 0), 2)

    for i, (x, y) in enumerate(corners):
        cv2.circle(output, (x, y), 6, (0, 255, 0), -1)
        put_text(output, str(i), (x + 8, y - 8), (0, 255, 0), scale=0.5, thickness=1)


def _draw_fruits(
    output: np.ndarray,
    corners: np.ndarray,
    fruits: list[Fruit],
) -> None:
    matrix = inverse_warp_matrix(corners)
    show_radius = load().get("debug_radius", False)

    for fruit in fruits:
        center, radius = screen_circle(matrix, fruit.x, fruit.y, fruit.radius)
        color = (0, 255, 0) if fruit.confidence >= 50 else (0, 165, 255)

        cv2.circle(output, center, radius, color, 2)
        cv2.circle(output, center, 2, color, -1)

        label = f"{fruit.name} {fruit.confidence:.0f}%"
        if show_radius:
            label += f" r={fruit.radius / NORMALIZED_WIDTH:.3f}"

        origin = (center[0] - radius, max(12, center[1] - radius - 6))
        put_text(output, label, origin, color, scale=0.4, thickness=1)

    put_text(output, f"fruits: {len(fruits)}", (8, 80), (0, 255, 255))


def _smooth_corners(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    board_width = float(np.linalg.norm(current[1] - current[0]))
    if board_width <= 0:
        return current

    shift = float(np.linalg.norm(current - previous, axis=1).max())
    if shift > board_width * CORNER_JUMP_RATIO:
        return current

    smoothed = previous * (1.0 - CORNER_SMOOTH_ALPHA) + current * CORNER_SMOOTH_ALPHA
    return smoothed.astype(np.float32)


def _find_corners(frame: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_lower, frame_upper = BOARD_FRAME_HSV
    mask = cv2.inRange(hsv, frame_lower, frame_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 辺に直線を当てるので、間引かずに輪郭の全点を受け取る。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * MIN_BOX_AREA_RATIO

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        corners = _contour_to_corners(contour)
        if corners is None or _touches_edge(corners, frame.shape[:2]):
            continue

        candidates.append((area, corners))

    if not candidates:
        return None

    _, corners = max(candidates, key=lambda item: item[0])
    return _inset_to_wall(_order_corners(corners))


def _inset_to_wall(corners: np.ndarray) -> np.ndarray:
    """検出した枠の外側の四隅を、本当の壁の位置まで内側へ寄せる。"""
    top_left, top_right, bottom_right, bottom_left = corners

    return np.array(
        [
            top_left
            + WALL_INSET_X_RATIO * (top_right - top_left)
            + WALL_INSET_Y_RATIO * (bottom_left - top_left),
            top_right
            + WALL_INSET_X_RATIO * (top_left - top_right)
            + WALL_INSET_Y_RATIO * (bottom_right - top_right),
            bottom_right
            + WALL_INSET_X_RATIO * (bottom_left - bottom_right)
            + WALL_INSET_Y_RATIO * (top_right - bottom_right),
            bottom_left
            + WALL_INSET_X_RATIO * (bottom_right - bottom_left)
            + WALL_INSET_Y_RATIO * (top_left - bottom_left),
        ],
        dtype=np.float32,
    )


def _touches_edge(corners: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = shape
    margin = max(1.0, min(height, width) * EDGE_MARGIN_RATIO)

    return bool(
        corners[:, 0].min() < margin
        or corners[:, 1].min() < margin
        or corners[:, 0].max() > width - 1 - margin
        or corners[:, 1].max() > height - 1 - margin
    )


def _contour_to_corners(contour: np.ndarray) -> np.ndarray | None:
    coarse = _coarse_corners(contour)
    if coarse is None:
        return None

    return _fit_corners(contour, _order_corners(coarse))


def _coarse_corners(contour: np.ndarray) -> np.ndarray | None:
    """どの点がどの辺に属するかを決めるための、大まかな四角形。"""
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)


def _fit_corners(contour: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """四辺に直線を当て、その交点を四隅とする。

    盤面の角は丸い。approxPolyDP の頂点は円弧の上に乗り、しかも許容誤差が
    周長の 2% (実測で 30px 超) あるので、頂点は真の角より内側に来る。
    minAreaRect も遠近で歪んだ盤面には合わない。辺は丸みの影響を受けない
    ので、辺から角を復元する。
    """
    points = contour.reshape(-1, 2).astype(np.float32)

    starts = coarse
    edges = np.roll(coarse, -1, axis=0) - starts
    lengths = np.linalg.norm(edges, axis=1)
    if (lengths <= 1e-6).any():
        return coarse

    units = edges / lengths[:, None]
    offsets = points[:, None, :] - starts[None, :, :]
    along = (offsets * units[None, :, :]).sum(axis=2)
    across = np.abs(
        offsets[:, :, 0] * units[None, :, 1] - offsets[:, :, 1] * units[None, :, 0]
    )
    nearest = np.argmin(across, axis=1)

    lines = []
    for side in range(4):
        skip = lengths[side] * CORNER_ARC_SKIP_RATIO
        on_side = (
            (nearest == side)
            & (along[:, side] > skip)
            & (along[:, side] < lengths[side] - skip)
        )
        if int(on_side.sum()) < MIN_SIDE_POINTS:
            return coarse

        lines.append(cv2.fitLine(points[on_side], cv2.DIST_L2, 0, 0.01, 0.01).ravel())

    fitted = []
    for side in range(4):
        corner = _intersect(lines[side - 1], lines[side])
        if corner is None:
            return coarse
        fitted.append(corner)

    corners = np.array(fitted, dtype=np.float32)

    # 辺の一部しか写っていないと直線が転び、交点が遠くへ飛ぶ。
    if np.linalg.norm(corners - coarse, axis=1).max() > lengths.max() * MAX_CORNER_SHIFT_RATIO:
        return coarse

    return corners


def _intersect(first: np.ndarray, second: np.ndarray) -> tuple[float, float] | None:
    """fitLine が返す (方向ベクトル, 通過点) 同士の交点。"""
    ax, ay, apx, apy = first
    bx, by, bpx, bpy = second

    denominator = ax * by - ay * bx
    if abs(denominator) < 1e-6:
        return None

    step = ((bpx - apx) * by - (bpy - apy) * bx) / denominator
    return float(apx + ax * step), float(apy + ay * step)


def _order_corners(corners: np.ndarray) -> np.ndarray:
    corners = np.array(corners, dtype=np.float32)
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).reshape(-1)

    top_left = corners[np.argmin(sums)]
    bottom_right = corners[np.argmax(sums)]
    top_right = corners[np.argmin(diffs)]
    bottom_left = corners[np.argmax(diffs)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _warp(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(
        frame,
        warp_matrix(corners),
        (NORMALIZED_WIDTH, NORMALIZED_HEIGHT),
    )
