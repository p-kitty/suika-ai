"""The REINFORCE advantage must not be the move number.

A batch-wide baseline made the advantage correlate -0.766 with the move number and left the gradient as noise
(NOTES 'Measured: REINFORCE on the ranker made it worse'). These pin the property that fixed it, on synthetic
rollouts, without playing the sim.
"""

from __future__ import annotations

import numpy as np

from scripts.train_rl import Rollout, _episode_grads, _split_half
from src.training.features import FEATURE_DIM


def _rollouts(n: int, length: int, rng: np.random.Generator) -> list[Rollout]:
    """Every game scores the same per move, so return-to-go depends only on the move number."""
    return [
        Rollout(seed=i, score=float(length), steps=length,
                grads=rng.normal(size=(length, FEATURE_DIM)),
                rewards=np.ones(length))
        for i in range(n)
    ]


def test_move_baseline_removes_what_only_the_move_number_explains() -> None:
    rng = np.random.default_rng(0)
    rows = _rollouts(16, 50, rng)
    # Identical reward streams: nothing depends on the choice, so the per-move advantage is zero everywhere.
    assert np.allclose(_episode_grads(rows, "move"), 0.0)
    # The batch baseline still hands out large advantages by move number alone.
    assert np.abs(_episode_grads(rows, "batch")).max() > 1e-3


def test_split_half_agrees_on_signal_and_not_on_noise() -> None:
    rng = np.random.default_rng(1)
    direction = rng.normal(size=FEATURE_DIM)
    signal = np.array([direction + 0.1 * rng.normal(size=FEATURE_DIM) for _ in range(64)])
    assert _split_half(signal, rng) > 0.9
    # Fresh noise per trial: resplitting one fixed sample only measures that sample's chance structure.
    noise = [_split_half(rng.normal(size=(64, FEATURE_DIM)), rng) for _ in range(200)]
    assert abs(float(np.median(noise))) < 0.1
