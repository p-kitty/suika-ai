"""落とす → 待つ → 読む。学習ループの 1 ステップ。"""

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
    # 狙い列 (正規化座標)。
    target_x: float | None
    # ダイアログで盤面が隠れたときだけ打ち切り。静止待ち timeout は done にしない。
    done: bool
    info: str


class Env:
    """画面を読んで、列を指定して落として、止まった盤面を返す。"""

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
            # Tracker は表示用。静止判定は平滑化前の生座標を残す。
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
        """盤面が止まってから列を決め、視線を合わせて落とす。

        x を省略し choose を渡すと、静止確認後の同じ観測で列を決める。
        動いている盤面を読んで狙うずれを防ぐ。
        abort が真を返したら操作を打ち切る (自動モードの停止用)。
        on_aim は狙い列が決まった直後 (視点移動の前) に呼ばれる。
        """
        before = self.observe()
        if before.blocked:
            return StepResult(before, None, done=True, info="dialog")

        # 自動プレイで ready 直後に落とすと、まだ連鎖中のことがある。
        before = settle.wait_playable(self.observe, abort=abort)
        if abort is not None and abort():
            return StepResult(before, None, done=False, info="aborted")
        if before.blocked:
            return StepResult(before, None, done=True, info="dialog")
        if not before.ready:
            return StepResult(before, None, done=False, info="not settled")

        if x is None:
            if choose is None:
                raise ValueError("x か choose が必要")
            x = choose(before)
        target = clamp_drop_x(x, before.held_type)
        if on_aim is not None:
            on_aim(target)
        read = self._aim_read

        # いま止まった観測で決めた列へすぐ狙う (再度 wait して盤面を取り直さない)。
        aimed = control.drop_column(target, read=read, abort=abort)
        if abort is not None and abort():
            # クリック前に止まったか、落とした直後かは drop_column 側で分岐済み。
            return StepResult(self.observe(), target, done=False, info="aborted")
        info_aim = "ok" if aimed else "aim_timeout"

        # 落としたあと、いったん held が消えるのを待つ。消えないまま静止判定に
        # 入ると、雲が動いただけの揺れで止まってしまう。
        _wait_held_gone(self.observe, before.held_x, abort=abort)

        after = settle.wait_playable(self.observe, abort=abort)
        if abort is not None and abort():
            return StepResult(after, target, done=False, info="aborted")
        # 静止待ちのタイムアウトは一時的な失敗。ダイアログだけ打ち切りにする。
        if after.blocked:
            return StepResult(after, target, done=True, info="dialog")
        if not after.ready:
            return StepResult(after, target, done=False, info="timeout")
        info = info_aim if not aimed else "ok"

        # 極端な端に居残らない程度に内側へ戻す (毎回中央までは戻さない)。
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
    raise RuntimeError("画面が取れない")


def _wait_held_gone(
    read: Callable[[], Observation],
    previous_x: float | None,
    timeout_sec: float = 2.0,
    abort: Callable[[], bool] | None = None,
) -> None:
    """落下待ちが消えるか、列が大きく動くまで待つ。

    クリックが効いているサイン。効いていなければそのまま静止待ちに進む。
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
