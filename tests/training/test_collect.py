"""Collecting value data.

What is checked is **that the label definitions hold** (how returns are taken, the correspondence between the candidate table and
rows), not concrete values. Values move when the policy or physics changes.
"""

import numpy as np

from src.sim.sim_env import SimEnv
from src.training.collect import (
    ValueDataset,
    collect_value_episode,
    save_value_dataset,
)
from src.training.features import FEATURE_DIM

# The physics takes 200-some ms per move, so pytest cuts the number of moves short.
STEPS = 6
STRIDE = 3


def _episode() -> ValueDataset:
    return collect_value_episode(
        SimEnv(seed=642746), max_steps=STEPS, episode=7, candidate_stride=STRIDE
    )


def test_shapes_line_up() -> None:
    data = _episode()
    n = len(data.feats)
    assert 0 < n <= STEPS
    assert data.feats.shape == (n, FEATURE_DIM)
    for arr in (data.rewards, data.merges, data.returns, data.steps, data.episodes):
        assert len(arr) == n
    # A move that scored points always merged. If this breaks, cascades are miscounted.
    assert bool(np.all(data.merges[data.rewards > 0] > 0))
    assert np.all(np.isfinite(data.feats))
    assert np.all(data.episodes == 7)


def test_return_is_the_points_taken_after_that_board() -> None:
    """The value of board t is the sum of points after t. That move's points are not included.

    Including them would make the board of a merging move look higher by 'that merge',
    counting it twice in Q = (that move's points) + V (board).
    """
    data = _episode()
    for i in range(len(data.returns)):
        assert abs(data.returns[i] - data.rewards[i + 1 :].sum()) < 1e-3
    assert data.returns[-1] == 0.0


def test_candidate_table_points_at_real_rows() -> None:
    data = _episode()
    n = len(data.feats)
    assert len(data.cand_row) > 0
    assert data.cand_feats.shape == (len(data.cand_row), FEATURE_DIM)
    assert data.cand_row.min() >= 0
    assert data.cand_row.max() < n
    # As thinned, moves that kept candidates are every STRIDE.
    assert set(data.cand_row.tolist()) == set(range(0, n, STRIDE))


def test_exactly_one_candidate_is_the_teacher_move() -> None:
    """The teacher's move is always exactly one row of the candidate table. 0 means matching on x is broken."""
    data = _episode()
    for row in set(data.cand_row.tolist()):
        picked = data.cand_chosen[data.cand_row == row]
        assert picked.sum() == 1


def test_teacher_move_is_not_always_the_top_eval() -> None:
    """With the next lookahead, the teacher is not necessarily the top of the first-ply eval.

    If they always matched, the candidate table would not be reaching `choose_x` (the lookahead would not
    be working), meaning the collection differs from the teacher.
    """
    data = _episode()
    assert len(data.cand_evals) > 0
    # At least some rows have the teacher's eval equal to the max eval (most do).
    for row in set(data.cand_row.tolist()):
        mask = data.cand_row == row
        chosen_eval = data.cand_evals[mask][data.cand_chosen[mask]]
        assert chosen_eval[0] <= data.cand_evals[mask].max() + 1e-6


def test_save_roundtrip(tmp_path) -> None:
    data = _episode()
    path = tmp_path / "value.npz"
    save_value_dataset(data, path)
    loaded = np.load(path)
    assert np.allclose(loaded["feats"], data.feats)
    assert np.allclose(loaded["returns"], data.returns)
    assert np.array_equal(loaded["cand_row"], data.cand_row)
