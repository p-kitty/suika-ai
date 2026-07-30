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

from .config import load
from .vision.normalized import NORMALIZED_WIDTH

# Win32 のマウス入力。
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

CLICK_PAUSE_SEC = 0.05
# 動かしたあと、視点と検出が落ち着くまで待つ。
LOOK_PAUSE_SEC = 0.09

# 狙いをまたいだあと、この幅以内なら打ち返さず止める。
CROSS_STOP = 14.0


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
    read: Callable[[], tuple[object, np.ndarray | None]],
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
    read: Callable[[], tuple[object, np.ndarray | None]],
    abort: Callable[[], bool] | None = None,
) -> bool:
    """次の手の前に、落下待ちを盤面中央へ戻す。クリックはしない。"""
    cfg = load()
    # 中央は厳密でなくてよい。寄せ切れず左右しないことを優先。
    tolerance = float(cfg.get("recenter_tolerance", 14))
    return aim(NORMALIZED_WIDTH / 2, read, tolerance=tolerance, abort=abort)


def aim(
    target_x: float,
    read: Callable[[], tuple[object, np.ndarray | None]],
    *,
    tolerance: float | None = None,
    abort: Callable[[], bool] | None = None,
) -> bool:
    """held_x を target_x 付近まで寄せる。行き過ぎたら打ち返さず止める。

    比例で寄せる。わざと少し短めに動かし、行き過ぎより手前止まりを優先する。
    """
    cfg = load()
    # 誤差 (盤面 px) に対するマウス移動。逆方向なら負にする。
    gain = float(cfg.get("look_gain", 0.55))
    if tolerance is None:
        tolerance = float(cfg.get("look_tolerance", 8))
    timeout_sec = float(cfg.get("look_timeout_sec", 4.0))
    max_step = int(cfg.get("look_max_step", 48))

    deadline = time.monotonic() + timeout_sec
    previous_error: float | None = None
    best_error: float | None = None
    stall_moves = 0

    while time.monotonic() < deadline:
        if abort is not None and abort():
            return False
        obs, _corners = read()
        if getattr(obs, "blocked", False):
            return False
        held_x = getattr(obs, "held_x", None)
        if held_x is None:
            time.sleep(LOOK_PAUSE_SEC)
            continue

        error = target_x - float(held_x)
        abs_error = abs(error)
        if abs_error <= tolerance:
            return True

        # 端など、これ以上 held が寄らないときは動かし続けず今の位置で落とす。
        if best_error is None or abs_error < best_error - 1.0:
            best_error = abs_error
            stall_moves = 0
        else:
            stall_moves += 1
            if stall_moves >= 3:
                return True

        # 狙いをまたいだ。近いなら打ち返さず採用。遠いときだけ弱く戻す。
        crossed = previous_error is not None and error * previous_error < 0
        if crossed and abs_error <= max(tolerance, CROSS_STOP):
            return True

        # 0.8 で手前に寄せる。近いほど一歩を抑える。
        scale = 0.3 if crossed else 0.8
        if abs_error < 24:
            scale *= 0.55
        raw = error * gain * scale
        magnitude = min(max_step, max(1, int(round(abs(raw)))))
        step = magnitude if raw > 0 else -magnitude

        move_by(step, 0)
        previous_error = error
        time.sleep(LOOK_PAUSE_SEC)

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
