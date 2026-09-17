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


def test_signal_noise_predicts_agreement_between_independent_batches() -> None:
    """The estimate from one batch must match what fresh, non-overlapping batches actually do."""
    from scripts.train_rl import _cos, _predicted_agree, _signal_noise

    rng = np.random.default_rng(2)
    direction = rng.normal(size=FEATURE_DIM)
    direction *= 0.3 / np.linalg.norm(direction)

    def batch(n: int) -> np.ndarray:
        return direction + rng.normal(size=(n, FEATURE_DIM))

    signal, noise = _signal_noise(batch(512))
    assert abs(signal - 0.09) < 0.06
    assert abs(noise - FEATURE_DIM) < 2.0
    for half in (16, 64):
        actual = np.mean([_cos(batch(half).sum(0), batch(half).sum(0)) for _ in range(300)])
        assert abs(_predicted_agree(half, signal, noise) - actual) < 0.1


def test_signal_noise_reports_no_signal_for_pure_noise() -> None:
    from scripts.train_rl import _signal_noise

    rng = np.random.default_rng(3)
    signal, noise = _signal_noise(rng.normal(size=(512, FEATURE_DIM)))
    assert abs(signal) < 0.1 * noise / 8


def test_saved_rollouts_reproduce_the_same_gradients(tmp_path) -> None:
    """--sweep must see exactly what --check-gradient played, or offline results mean nothing."""
    from scripts.train_rl import _load_rows, _save_rows

    rng = np.random.default_rng(4)
    rows = [Rollout(seed=i, score=float(n), steps=n, grads=rng.normal(size=(n, FEATURE_DIM)),
                    rewards=rng.integers(0, 20, size=n).astype(float))
            for i, n in enumerate((30, 55, 12))]
    path = tmp_path / "rollouts.npz"
    _save_rows(path, rows)
    back = _load_rows(path)
    for baseline in ("batch", "move", "move-mean"):
        assert np.allclose(_episode_grads(rows, baseline, 0.995), _episode_grads(back, baseline, 0.995), atol=1e-4)
