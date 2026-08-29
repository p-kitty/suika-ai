"""価値データの収集。

見るのは**ラベルの定義が保たれていること**（リターンの取り方、候補表と行の
対応）で、具体的な値ではない。値は方策と物理が変われば動く。
"""

import numpy as np

from src.sim.sim_env import SimEnv
from src.training.collect import (
    ValueDataset,
    collect_value_episode,
    save_value_dataset,
)
from src.training.features import FEATURE_DIM

# 物理が 1 手 200ms 台なので、pytest では手数を切り詰める。
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
    # 点が付いた手は必ず合体している。ここが崩れると cascades を数え違える。
    assert bool(np.all(data.merges[data.rewards > 0] > 0))
    assert np.all(np.isfinite(data.feats))
    assert np.all(data.episodes == 7)


def test_return_is_the_points_taken_after_that_board() -> None:
    """盤 t の値は t より後の点の総和。その手の点は含めない。

    ここを含めると、合体した手の盤が「その合体ぶん」だけ高く見え、
    Q =（その手の点）+ V（盤）で二重に数えることになる。
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
    # 間引きの通り、候補を残した手は STRIDE おき。
    assert set(data.cand_row.tolist()) == set(range(0, n, STRIDE))


def test_exactly_one_candidate_is_the_teacher_move() -> None:
    """教師の手は必ず候補表の 1 本。0 本なら x の突き合わせが壊れている。"""
    data = _episode()
    for row in set(data.cand_row.tolist()):
        picked = data.cand_chosen[data.cand_row == row]
        assert picked.sum() == 1


def test_teacher_move_is_not_always_the_top_eval() -> None:
    """next 先読みがあるので、教師は 1 手目 eval の最上位とは限らない。

    ここが常に一致するなら `choose_x` に候補表が渡っていない（先読みが
    効いていない）ということなので、収集が教師と別物になっている。
    """
    data = _episode()
    assert len(data.cand_evals) > 0
    # 少なくとも eval の最大値と教師の eval が一致する行はある（大半はそう）。
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
