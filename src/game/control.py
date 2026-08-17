"""一人称視点を動かして狙い、クリックで落とす。

VRChat のスイカは画面クリックではなく、FPS と同じくマウス相対移動で視線を動かす。
狙いは落下待ちの列 (held_x) を狙い列に重ねること。盤面座標で誤差を見る
(画面射影は四隅の揺らぎで左右に振れやすい)。
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable
from ctypes import wintypes

import numpy as np

from ..observe import Observation
from ..vision.normalized import NORMALIZED_WIDTH

# Win32 のマウス入力。
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

CLICK_PAUSE_SEC = 0.05
# 動かしたあと、視点と検出が落ち着くまで待つ。
LOOK_PAUSE_SEC = 0.09

# 誤差 (盤面 px) に対するマウス移動。わざと少し短めに寄せる。
LOOK_GAIN = 0.55
# 盤面幅 400 に対し、チェリー半径 (~12) の 1/3 程度。8 だと列がぶれる。
LOOK_TOLERANCE = 4.0
LOOK_TIMEOUT_SEC = 4.0
LOOK_MAX_STEP = 48
# 狙いをまたいだあと、この幅以内なら打ち返さず止める。
CROSS_STOP = 10.0
# 端付近は held が壁で止まり、検出も揺れる。厳密に寄せようとして視点だけが
# 振れ続けるのを避ける。
EDGE_BAND = 48.0
EDGE_TOLERANCE = 18.0
# held がほとんど動かないのに視点だけ進むのを何手で諦めるか。
STALL_MOVES = 2
# 落下後は毎回盤面中央へ戻す。端に居残ると次手の視点がズレやすい。
RECENTER_X = NORMALIZED_WIDTH / 2
RECENTER_TOLERANCE = LOOK_TOLERANCE


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    _anonymous_ = ("i",)
    _fields_ = [("type", wintypes.DWORD), ("i", _I)]


def drop_column(
    target_x: float,
    *,
    read: Callable[[], tuple[Observation, np.ndarray | None]],
    abort: Callable[[], bool] | None = None,
) -> bool:
    """落下待ちの列を target_x に重ねてからクリックする。"""
    aimed = aim(target_x, read, abort=abort)
    # 狙い中／直後に中断されたら落とさない。
    if abort is not None and abort():
        return False
    click()
    return aimed


def recenter(
    read: Callable[[], tuple[Observation, np.ndarray | None]],
    abort: Callable[[], bool] | None = None,
) -> bool:
    """次の手の前に、視線を盤面中央へ戻す。クリックはしない。"""
    if abort is not None and abort():
        return False
    obs, _corners = read()
    if obs.blocked:
        return False
    if obs.held_x is None:
        return False

    held = float(obs.held_x)
    if abs(held - RECENTER_X) <= RECENTER_TOLERANCE:
        return True
    return aim(RECENTER_X, read, tolerance=RECENTER_TOLERANCE, abort=abort)


def aim(
    target_x: float,
    read: Callable[[], tuple[Observation, np.ndarray | None]],
    *,
    tolerance: float | None = None,
    abort: Callable[[], bool] | None = None,
) -> bool:
    """held_x を target_x 付近まで寄せる。行き過ぎたら打ち返さず止める。

    比例で寄せる。わざと少し短めに動かし、行き過ぎより手前止まりを優先する。
    """
    # 誤差 (盤面 px) に対するマウス移動。逆方向なら負にする。
    if tolerance is None:
        tolerance = LOOK_TOLERANCE

    deadline = time.monotonic() + LOOK_TIMEOUT_SEC
    previous_error: float | None = None
    previous_held: float | None = None
    best_error: float | None = None
    stall_moves = 0
    no_improve = 0

    while time.monotonic() < deadline:
        if abort is not None and abort():
            return False
        obs, _corners = read()
        if obs.blocked:
            return False
        if obs.held_x is None:
            time.sleep(LOOK_PAUSE_SEC)
            continue

        held = float(obs.held_x)
        error = target_x - held
        abs_error = abs(error)
        if abs_error <= tolerance:
            return True
        # 右端・左端: 壁際まで来ていれば、残り誤差のために視点を振らない。
        if _edge_close_enough(target_x, held, tolerance):
            return True

        # held が動かない = 壁などで止まっている。視点だけ進めても無駄。
        if previous_held is not None and abs(held - previous_held) < 0.5:
            stall_moves += 1
            if stall_moves >= STALL_MOVES:
                return True
        else:
            stall_moves = 0

        # 誤差が改善しないときも打ち切る (検出揺れで微改善し続ける対策)。
        if best_error is None or abs_error < best_error - 2.0:
            best_error = abs_error
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                return True

        # 狙いをまたいだ。近いなら打ち返さず採用。遠いときだけ弱く戻す。
        crossed = previous_error is not None and error * previous_error < 0
        if crossed and abs_error <= max(tolerance, CROSS_STOP):
            return True

        # 0.8 で手前に寄せる。近いほど一歩を抑える。端は一歩をさらに抑える。
        scale = 0.3 if crossed else 0.8
        if abs_error < 24:
            scale *= 0.55
        if _near_edge(target_x) or _near_edge(held):
            scale *= 0.5
        raw = error * LOOK_GAIN * scale
        magnitude = min(LOOK_MAX_STEP, max(1, int(round(abs(raw)))))
        step = magnitude if raw > 0 else -magnitude

        move_by(step, 0)
        previous_error = error
        previous_held = held
        time.sleep(LOOK_PAUSE_SEC)

    return False


def _near_edge(x: float) -> bool:
    return x <= EDGE_BAND or x >= NORMALIZED_WIDTH - EDGE_BAND


def _edge_close_enough(target_x: float, held_x: float, tolerance: float) -> bool:
    """壁際の狙いで、held が同じ側の端まで来ていれば十分とみなす。

    EDGE_BAND 全体だと「端チェリーの左隣」みたいな内側狙いまで端扱いになり、
    held が壁際に居残ったまま落ちる (肩 → 対岸まで弾かれる)。
    本当に壁際の列だけを対象にする。
    """
    # 壁そのもの付近。EDGE_BAND (48) より狭い。
    wall = max(EDGE_TOLERANCE * 2, 28.0)
    near_wall = target_x >= NORMALIZED_WIDTH - wall or target_x <= wall
    if not near_wall:
        return False
    limit = max(tolerance, EDGE_TOLERANCE)
    if abs(target_x - held_x) <= limit:
        return True
    # 右端狙い: held が目標以上に右へ来ている / 左端は対称。
    if target_x >= NORMALIZED_WIDTH - wall and held_x >= target_x - limit:
        return True
    if target_x <= wall and held_x <= target_x + limit:
        return True
    return False


def move_by(dx: int, dy: int = 0) -> None:
    """マウスを相対移動する (FPS の視点操作)。"""
    if dx == 0 and dy == 0:
        return
    _send(MOUSEEVENTF_MOVE, int(dx), int(dy))


def click() -> None:
    """今の視線のまま左クリック。"""
    _send(MOUSEEVENTF_LEFTDOWN)
    time.sleep(CLICK_PAUSE_SEC)
    _send(MOUSEEVENTF_LEFTUP)


def _send(flags: int, dx: int = 0, dy: int = 0) -> None:
    event = INPUT(type=INPUT_MOUSE)
    event.mi = MOUSEINPUT(dx, dy, 0, flags, 0, None)
    sent = ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event))
    if sent != 1:
        raise RuntimeError("SendInput が失敗した")
