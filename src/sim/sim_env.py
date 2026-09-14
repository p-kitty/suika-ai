"""A headless drop simulator. For offline learning."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np

from ..observe import Observation, clamp_drop_x
from ..policy import drop_scores
from ..reward import cleared_double_watermelon, is_game_over
from ..vision.colors import SPAWN_MAX_TYPE
from ..vision.normalized import NORMALIZED_WIDTH
from ..vision.state import Fruit

# Half-width (board px) of the uniform error between the chosen column and where the fruit is released.
# The sim drops exactly where choose_x aims, while the real game stops aiming anywhere inside
# control.LOOK_TOLERANCE (4px), up to CROSS_STOP (10px) after overshooting, and EDGE_TOLERANCE (18px)
# near the walls. 0 keeps the sim exact; A/B scripts rewrite it to measure how much that error costs.
AIM_NOISE_PX = 0.0


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
        # Even with seed=None, fix it to a concrete value. This env keeps no record
        # of the run anywhere, so if the seed cannot be stated afterwards a broken game cannot be replayed
        # (view_sim shows it in the footer). The digit count is kept short enough to read from a screenshot.
        # The way random seeds are drawn matches compare_policy.py.
        self.seed = secrets.randbelow(1_000_000) if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        # A separate stream, so turning the aim error on does not change the draws of the same seed
        # (A and B stay paired on the same fruit sequence).
        self._aim_rng = np.random.default_rng([self.seed, 1])
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
        if AIM_NOISE_PX > 0.0:
            x += float(self._aim_rng.uniform(-AIM_NOISE_PX, AIM_NOISE_PX))
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
