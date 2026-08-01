"""A headless drop simulator. For offline learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observe import Observation, clamp_drop_x
from .policy import drop_scores
from .reward import cleared_double_watermelon, is_game_over
from .vision.colors import SPAWN_MAX_TYPE
from .vision.normalized import NORMALIZED_WIDTH
from .vision.state import Fruit


@dataclass
class SimStep:
    observation: Observation
    # The real game's merge points. The learning reward is this.
    score: float
    done: bool
    merges: int
    info: str


class SimEnv:
    """Deal held/next from an empty board and drop with sim_physics.simulate_drop."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.fruits: list[Fruit] = []
        self.held_type: int | None = None
        self.next_type: int | None = None

    def reset(self) -> Observation:
        self.fruits = []
        self.held_type = self._spawn()
        self.next_type = self._spawn()
        return self._obs()

    def step(self, x: float) -> SimStep:
        if self.held_type is None:
            raise RuntimeError("reset has not been called")

        before = self._obs()
        target = clamp_drop_x(x, self.held_type)
        score, _penalties, _eval, after_fruits, merges = drop_scores(
            self.fruits, self.held_type, target
        )
        self.fruits = after_fruits
        self.held_type = self.next_type
        self.next_type = self._spawn()
        after = self._obs()

        dead = is_game_over(after)
        win = cleared_double_watermelon(before, after, merges=merges)
        done = dead or win
        if win:
            info = "win"
        elif dead:
            info = "dead"
        else:
            info = "ok"
        return SimStep(after, score, done, merges, info)

    def _spawn(self) -> int:
        return int(self.rng.integers(0, SPAWN_MAX_TYPE + 1))

    def _obs(self) -> Observation:
        held = self.held_type
        return Observation(
            ready=held is not None,
            blocked=False,
            fruits=tuple(self.fruits),
            held_type=held,
            held_x=NORMALIZED_WIDTH / 2 if held is not None else None,
            next_type=self.next_type,
        )
