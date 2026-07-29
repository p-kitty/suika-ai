"""落とす → 待つ → 読む。学習ループの 1 ステップ。"""

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
    # 狙い列 (正規化座標)。dry_run でも入る。
    target_x: float | None
    # ダイアログで盤面が隠れた、またはタイムアウトで ready に戻れなかった。
    done: bool
    info: str


class Env:
    """画面を読んで、列を指定して落として、止まった盤面を返す。"""

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
        """列 x (正規化座標) に視線を合わせて落とし、次の観測を返す。"""
        before = self.observe()
        if before.blocked:
            return StepResult(before, None, done=True, info="dialog")
        if not before.ready:
            return StepResult(before, None, done=False, info="not ready")

        target = clamp_drop_x(x, before.held_type)
        read = self._aim_read

        if self.dry_run:
            return StepResult(before, target, done=False, info="dry_run")

        aimed = control.drop_column(target, read=read, dry_run=False)
        info_aim = "ok" if aimed else "aim_timeout"

        # 落としたあと、いったん held が消えるのを待つ。消えないまま静止判定に
        # 入ると、雲が動いただけの揺れで止まってしまう。
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

        # 次の手が端から始まらないよう、新しい落下待ちを中央へ戻す。
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
    raise RuntimeError("画面が取れない")


def _wait_held_gone(read, previous_x: float | None, timeout_sec: float = 2.0) -> None:
    """落下待ちが消えるか、列が大きく動くまで待つ。

    クリックが効いているサイン。効いていなければそのまま静止待ちに進む。
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
