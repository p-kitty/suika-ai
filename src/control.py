"""Aim by moving the first-person view, and drop with a click.

Suika in VRChat moves the view by relative mouse movement like an FPS, not by clicking on the screen.
Aiming means lining up the column of the waiting fruit (held_x) with the target column. The error is measured in board coordinates
(screen projection tends to swing left and right with jitter of the four corners).
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable
from ctypes import wintypes

import numpy as np

from .observe import Observation
from .vision.normalized import NORMALIZED_WIDTH

# Win32 mouse input.
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

CLICK_PAUSE_SEC = 0.05
# After moving, wait until the view and detection settle.
LOOK_PAUSE_SEC = 0.09

# Mouse movement per unit of error (board px). Deliberately a little short.
LOOK_GAIN = 0.55
# About 1/3 of the cherry radius (~12) against a board width of 400. At 8 the column wobbles.
LOOK_TOLERANCE = 4.0
LOOK_TIMEOUT_SEC = 4.0
LOOK_MAX_STEP = 48
# After crossing the target, stop without correcting back if within this width.
CROSS_STOP = 10.0
# Near the edges held stops at the wall and detection also wobbles. Avoid the view
# swinging forever while trying to line up exactly.
EDGE_BAND = 48.0
EDGE_TOLERANCE = 18.0
# After how many moves to give up when only the view advances while held barely moves.
STALL_MOVES = 2
# Return to the board center after every drop. Staying at an edge tends to offset the view for the next move.
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
    """Line up the waiting column with target_x, then click."""
    aimed = aim(target_x, read, abort=abort)
    # Do not drop if interrupted during or right after aiming.
    if abort is not None and abort():
        return False
    click()
    return aimed


def recenter(
    read: Callable[[], tuple[Observation, np.ndarray | None]],
    abort: Callable[[], bool] | None = None,
) -> bool:
    """Return the view to the board center before the next move. Does not click."""
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
    """Bring held_x close to target_x. If it overshoots, stop without correcting back.

    Move proportionally. Deliberately move a little short, preferring stopping short over overshooting.
    """
    # Mouse movement per unit of error (board px). Negative for the opposite direction.
    if tolerance is None:
        tolerance = LOOK_TOLERANCE

    deadline = time.monotonic() + LOOK_TIMEOUT_SEC
    previous_error: float | None = None
    previous_held: float | None = None
    best_error: float | None = None
    stall_moves = 0

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
        # Right edge / left edge: once at the wall, do not swing the view for the remaining error.
        if _edge_close_enough(target_x, held, tolerance):
            return True

        # held does not move = stopped by a wall or similar. Advancing only the view is useless.
        if previous_held is not None and abs(held - previous_held) < 0.5:
            stall_moves += 1
            if stall_moves >= STALL_MOVES:
                return True
        else:
            stall_moves = 0

        # Also stop when the error does not improve (against detection wobble improving it slightly forever).
        if best_error is None or abs_error < best_error - 2.0:
            best_error = abs_error
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                return True

        # Crossed the target. If close, accept without correcting back. Only when far, go back gently.
        crossed = previous_error is not None and error * previous_error < 0
        if crossed and abs_error <= max(tolerance, CROSS_STOP):
            return True

        # Pull short at 0.8. The closer, the smaller the step. At the edges the step is reduced further.
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
    """When aiming at a wall column, consider it enough if held has reached the edge on the same side.

    The whole EDGE_BAND would treat inner aims such as 'just left of an edge cherry' as edges too,
    and held would drop while still sitting at the wall (shoulder → knocked to the far side).
    Only columns truly at the wall are targeted.
    """
    # Right at the wall itself. Narrower than EDGE_BAND (48).
    wall = max(EDGE_TOLERANCE * 2, 28.0)
    near_wall = target_x >= NORMALIZED_WIDTH - wall or target_x <= wall
    if not near_wall:
        return False
    limit = max(tolerance, EDGE_TOLERANCE)
    if abs(target_x - held_x) <= limit:
        return True
    # Right-edge aim: held has come further right than the target / the left edge is symmetric.
    if target_x >= NORMALIZED_WIDTH - wall and held_x >= target_x - limit:
        return True
    if target_x <= wall and held_x <= target_x + limit:
        return True
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


def _send(flags: int, dx: int = 0, dy: int = 0) -> None:
    event = INPUT(type=INPUT_MOUSE)
    event.mi = MOUSEINPUT(dx, dy, 0, flags, 0, None)
    sent = ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event))
    if sent != 1:
        raise RuntimeError("SendInput failed")
