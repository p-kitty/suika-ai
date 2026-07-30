"""画面なしの落下シミュレータ。オフライン学習用。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observe import Observation, clamp_drop_x
from .policy import simulate_drop
from .reward import is_game_over, step_reward
from .vision.colors import SPAWN_MAX_TYPE
from .vision.normalized import NORMALIZED_WIDTH
from .vision.state import Fruit


@dataclass
class SimStep:
    observation: Observation
    reward: float
    done: bool
    merges: int
    info: str


class SimEnv:
    """空盤から held/next を配り、policy.simulate_drop で落とす。"""

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
            raise RuntimeError("reset していない")

        before = self._obs()
        target = clamp_drop_x(x, self.held_type)
        after_fruits, merges = simulate_drop(self.fruits, self.held_type, target)
        self.fruits = after_fruits
        self.held_type = self.next_type
        self.next_type = self._spawn()
        after = self._obs()

        done = is_game_over(after)
        reward = step_reward(before, after, merges=merges, done=done)
        return SimStep(after, reward, done, merges, "ok" if not done else "dead")

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
