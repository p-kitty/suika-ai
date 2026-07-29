"""Aim by moving the first-person view, and drop with a click.

Suika in VRChat moves the view by relative mouse movement like an FPS, not by clicking on the screen.
Aiming means lining up the column of the waiting fruit (held_x) with the target column. The error is measured in board coordinates
(screen projection tends to swing left and right with jitter of the four corners).

The OpenCV debug window eats input, so it is hidden during controls and VRChat is brought to the front.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable
from ctypes import wintypes

import numpy as np

from .config import load
from .vision.normalized import NORMALIZED_WIDTH

# Win32 mouse input.
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
SW_HIDE = 0
SW_SHOW = 5
SW_RESTORE = 9

VRCHAT_TITLE = "VRChat"
SUIKA_TITLE = "Suika"

CLICK_PAUSE_SEC = 0.05
# After moving, wait until the view and detection settle. Too short and it corrects back and looks around.
LOOK_PAUSE_SEC = 0.14

# Consider it reached when within tolerance consecutively.
OK_FRAMES = 2
# After crossing the target, stop without correcting back if within this width.
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
    dry_run: bool = False,
) -> bool:
    """Line up the waiting column with target_x, then click."""
    if dry_run:
        return True

    with hidden(SUIKA_TITLE):
        focus(VRCHAT_TITLE)
        aimed = aim(target_x, read)
        click()
        return aimed


def recenter(
    read: Callable[[], tuple[object, np.ndarray | None]],
) -> bool:
    """Before the next move, return the waiting fruit to the board center. Does not click."""
    cfg = load()
    # The center need not be exact. Prefer not wobbling left and right without reaching it.
    tolerance = float(cfg.get("recenter_tolerance", 14))
    with hidden(SUIKA_TITLE):
        focus(VRCHAT_TITLE)
        return aim(NORMALIZED_WIDTH / 2, read, tolerance=tolerance)


def aim(
    target_x: float,
    read: Callable[[], tuple[object, np.ndarray | None]],
    *,
    tolerance: float | None = None,
) -> bool:
    """Bring held_x close to target_x. If it overshoots, stop without correcting back."""
    cfg = load()
    # Mouse movement per unit of error (board px). Negative for the opposite direction.
    gain = float(cfg.get("look_gain", 0.4))
    if tolerance is None:
        tolerance = float(cfg.get("look_tolerance", 8))
    timeout_sec = float(cfg.get("look_timeout_sec", 4.0))
    max_step = int(cfg.get("look_max_step", 24))

    deadline = time.monotonic() + timeout_sec
    previous_error: float | None = None
    ok_frames = 0
    best_error: float | None = None
    stall_moves = 0

    while time.monotonic() < deadline:
        obs, _corners = read()
        if getattr(obs, "blocked", False):
            return False
        held_x = getattr(obs, "held_x", None)
        if held_x is None:
            time.sleep(LOOK_PAUSE_SEC)
            continue

        error = target_x - float(held_x)
        if abs(error) <= tolerance:
            ok_frames += 1
            if ok_frames >= OK_FRAMES:
                return True
            time.sleep(LOOK_PAUSE_SEC)
            continue
        ok_frames = 0

        # When held will not move any further, at an edge and so on, do not keep moving; drop at the current position.
        abs_error = abs(error)
        if best_error is None or abs_error < best_error - 1.0:
            best_error = abs_error
            stall_moves = 0
        else:
            stall_moves += 1
            if stall_moves >= 3:
                return True

        # Crossed the target. If close, accept without correcting back. Only when far, go back gently.
        crossed = previous_error is not None and error * previous_error < 0
        if crossed and abs_error <= max(tolerance, CROSS_STOP):
            return True

        scale = 0.25 if crossed else 1.0
        raw = error * gain * scale
        # sqrt so it does not swing too far. The closer, the smaller the step.
        magnitude = min(max_step, max(1, int(round((abs(raw) ** 0.5) * 2.5))))
        step = magnitude if raw > 0 else -magnitude

        move_by(step, 0)
        previous_error = error
        time.sleep(LOOK_PAUSE_SEC)

    return False


def move_by(dx: int, dy: int = 0) -> None:
    """Move the mouse relatively (FPS view control)."""
    if dx == 0 and dy == 0:
        return
    _send(MOUSEEVENTF_MOVE, int(dx), int(dy))


def click() -> None:
    """Left-click with the current view."""
    _send(MOUSEEVENTF_LEFTDOWN)
    time.sleep(CLICK_PAUSE_SEC)
    _send(MOUSEEVENTF_LEFTUP)


def focus(title: str) -> None:
    """Bring the window with the given title to the front."""
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    if not hwnd:
        return
    ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    time.sleep(0.05)


def _send(flags: int, dx: int = 0, dy: int = 0) -> None:
    event = INPUT(type=INPUT_MOUSE)
    event.mi = MOUSEINPUT(dx, dy, 0, flags, 0, None)
    sent = ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event))
    if sent != 1:
        raise RuntimeError("SendInput failed")


class hidden:
    """Hide the given window during controls."""

    def __init__(self, title: str | None) -> None:
        self.title = title
        self.hwnd = 0

    def __enter__(self) -> None:
        if not self.title:
            return
        self.hwnd = ctypes.windll.user32.FindWindowW(None, self.title)
        if self.hwnd:
            ctypes.windll.user32.ShowWindow(self.hwnd, SW_HIDE)
            time.sleep(0.05)

    def __exit__(self, *_exc) -> None:
        if self.hwnd:
            ctypes.windll.user32.ShowWindow(self.hwnd, SW_SHOW)
