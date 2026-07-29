"""Drop → wait → read. One step of the learning loop."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import control, settle
from .capture import capture
from .config import load
from .observe import Observation, clamp_drop_x, from_board
from .tracker import Tracker
from .vision.board import BoardResult, localize


@dataclass
class StepResult:
    observation: Observation
    # Target column (normalized coordinates). Set even in dry_run.
    target_x: float | None
    # A dialog hid the board, or it could not get back to ready due to a timeout.
    done: bool
    info: str


class Env:
    """Read the screen, drop at the given column, and return the settled board."""

    def __init__(self, *, dry_run: bool | None = None) -> None:
        self.tracker = Tracker()
        self.previous_corners: np.ndarray | None = None
        self._last_board: BoardResult | None = None
        if dry_run is None:
            dry_run = not load().get("control_enabled", False)
        self.dry_run = dry_run

    def reset(self) -> Observation:
        self.tracker.reset()
        self.previous_corners = None
        return self.observe()

    def observe(self, frame: np.ndarray | None = None) -> Observation:
        if frame is None:
            frame = _grab()
        result = localize(frame, self.previous_corners)
        self.previous_corners = result.corners

        if result.fruits is None:
            self.tracker.reset()
        else:
            result.fruits = self.tracker.update(result.fruits)

        self._last_board = result
        return from_board(result)

    @property
    def board(self) -> BoardResult | None:
        return self._last_board

    def step(self, x: float) -> StepResult:
        """Line up the view with column x (normalized coordinates), drop, and return the next observation."""
        before = self.observe()
        if before.blocked:
            return StepResult(before, None, done=True, info="dialog")
        if not before.ready:
            return StepResult(before, None, done=False, info="not ready")

        target = clamp_drop_x(x, before.held_type)
        read = self._aim_read

        if self.dry_run:
            return StepResult(before, target, done=False, info="dry_run")

        # Hide Suika during controls, settle wait and return to center, and bring it back once at the end.
        with control.hidden(control.SUIKA_TITLE):
            aimed = control.drop_column(target, read=read, dry_run=False)
            info_aim = "ok" if aimed else "aim_timeout"

            # After dropping, wait once for held to disappear. If it does not disappear, the settle check
            # stops on wobble that is only the clouds moving.
            _wait_held_gone(self.observe, before.held_x)

            settled = settle.wait_settled(self.observe)
            if settled.blocked:
                return StepResult(settled, target, done=True, info="dialog")

            ready = settle.wait_ready(self.observe)
            done = ready.blocked or not ready.ready
            if ready.blocked:
                info = "dialog"
            elif not ready.ready:
                info = "timeout"
            elif not aimed:
                info = info_aim
            else:
                info = "ok"

            # Return the new waiting fruit to center so the next move does not start from the edge.
            if not done and ready.ready:
                control.recenter(read)
                ready = self.observe()

        return StepResult(ready, target, done=done, info=info)

    def _aim_read(self):
        obs = self.observe()
        corners = None if self.board is None else self.board.corners
        return obs, corners


def _grab():
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        frame = capture()
        if frame is not None:
            return frame
        time.sleep(0.02)
    raise RuntimeError("cannot capture the screen")


def _wait_held_gone(read, previous_x: float | None, timeout_sec: float = 2.0) -> None:
    """Wait until the waiting fruit disappears or the column moves a lot.

    A sign that the click worked. If it did not, proceed straight to the settle wait.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        obs = read()
        if obs.blocked:
            return
        if not obs.ready:
            return
        if previous_x is not None and abs((obs.held_x or 0) - previous_x) > 30:
            return
        time.sleep(1 / 30)
