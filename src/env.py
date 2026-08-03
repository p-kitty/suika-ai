"""Drop → wait → read. One step of the learning loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import control, settle
from .capture import capture
from .observe import Observation, clamp_drop_x, from_board
from .tracker import Tracker
from .vision.board import BoardResult, localize
from .vision.state import Fruit


@dataclass
class StepResult:
    observation: Observation
    # Target column (normalized coordinates).
    target_x: float | None
    # Stop only when a dialog hides the board. A settle-wait timeout is not done.
    done: bool
    info: str


class Env:
    """Read the screen, drop at the given column, and return the settled board."""

    def __init__(self) -> None:
        self.tracker = Tracker()
        self.previous_corners: np.ndarray | None = None
        self._last_board: BoardResult | None = None

    def reset(self) -> Observation:
        self.tracker.reset()
        self.previous_corners = None
        return self.observe()

    def observe(self, frame: np.ndarray | None = None) -> Observation:
        if frame is None:
            frame = _grab()
        result = localize(frame, self.previous_corners)
        self.previous_corners = result.corners

        raw: list[Fruit] | None
        if result.fruits is None:
            self.tracker.reset()
            raw = None
        else:
            # Tracker is for display. The settle check keeps the raw coordinates before smoothing.
            raw = list(result.fruits)
            result.fruits = self.tracker.update(raw)

        self._last_board = result
        return from_board(result, raw_fruits=raw)

    @property
    def board(self) -> BoardResult | None:
        return self._last_board

    def step(
        self,
        x: float | None = None,
        abort: Callable[[], bool] | None = None,
        *,
        choose: Callable[[Observation], float] | None = None,
        on_aim: Callable[[float], None] | None = None,
    ) -> StepResult:
        """Decide the column after the board stops, line up the view, and drop.

        If x is omitted and choose is passed, the column is decided on the same observation after the settle check.
        Prevents the offset of aiming from a board that is still moving.
        If abort returns true, the operation is aborted (for stopping auto mode).
        on_aim is called right after the target column is decided (before the view moves).
        """
        before = self.observe()
        if before.blocked:
            return StepResult(before, None, done=True, info="dialog")

        # Dropping right after ready in auto play can still be mid-cascade.
        before = settle.wait_playable(self.observe, abort=abort)
        if abort is not None and abort():
            return StepResult(before, None, done=False, info="aborted")
        if before.blocked:
            return StepResult(before, None, done=True, info="dialog")
        if not before.ready:
            return StepResult(before, None, done=False, info="not settled")

        if x is None:
            if choose is None:
                raise ValueError("x or choose is required")
            x = choose(before)
        target = clamp_drop_x(x, before.held_type)
        if on_aim is not None:
            on_aim(target)
        read = self._aim_read

        # Aim right away at the column decided on the observation that just stopped (do not wait again to reread the board).
        aimed = control.drop_column(target, read=read, abort=abort)
        if abort is not None and abort():
            # Whether it stopped before the click or right after dropping is already branched in drop_column.
            return StepResult(self.observe(), target, done=False, info="aborted")
        info = "ok" if aimed else "aim_timeout"

        # After dropping, wait once for held to disappear. If it does not disappear, the settle check
        # stops on wobble that is only the clouds moving.
        _wait_held_gone(self.observe, before.held_x, abort=abort)

        after = settle.wait_playable(self.observe, abort=abort)
        if abort is not None and abort():
            return StepResult(after, target, done=False, info="aborted")
        # A settle-wait timeout is a temporary failure. Only a dialog aborts.
        if after.blocked:
            return StepResult(after, target, done=True, info="dialog")
        if not after.ready:
            return StepResult(after, target, done=False, info="timeout")

        # After dropping, return the view to the center so the next move's reference does not drift.
        control.recenter(read, abort=abort)
        after = self.observe()

        return StepResult(after, target, done=False, info=info)

    def _aim_read(self) -> tuple[Observation, np.ndarray | None]:
        obs = self.observe()
        corners = None if self.board is None else self.board.corners
        return obs, corners


def _grab() -> np.ndarray:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        frame = capture()
        if frame is not None:
            return frame
        time.sleep(0.02)
    raise RuntimeError("cannot capture the screen")


def _wait_held_gone(
    read: Callable[[], Observation],
    previous_x: float | None,
    timeout_sec: float = 2.0,
    abort: Callable[[], bool] | None = None,
) -> None:
    """Wait until the waiting fruit disappears or the column moves a lot.

    A sign that the click worked. If it did not, proceed straight to the settle wait.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if abort is not None and abort():
            return
        obs = read()
        if obs.blocked:
            return
        if not obs.ready:
            return
        if previous_x is not None and abs((obs.held_x or 0) - previous_x) > 30:
            return
        time.sleep(1 / 30)
