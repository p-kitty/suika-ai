"""落下後の盤の特徴。

見るのは**項が分かれて出てくること**と、角スイカの幾何が拾えること。
値そのものは `penalties.py` の重み次第で動くので固定しない。
"""

import numpy as np

from src import penalties as pen
from src.training.features import FEATURE_DIM, FEATURE_NAMES, board_features
from src.vision.classify import fruit_radius
from src.vision.colors import MAX_FRUIT_TYPE
from src.vision.normalized import NORMALIZED_HEIGHT
from src.vision.state import Fruit


def _fruit(fruit_type: int, x: float, y: float) -> Fruit:
    return Fruit(
        type=fruit_type, x=x, y=y, radius=fruit_radius(fruit_type), confidence=90
    )


def _value(vec: np.ndarray, name: str) -> float:
    return float(vec[FEATURE_NAMES.index(name)])


def test_empty_board_is_finite() -> None:
    vec = board_features([], sign=1)
    assert vec.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(vec))
    # 空盤は死から最も遠い。0 にすると「負けラインちょうど」と区別が付かない。
    assert _value(vec, "crown_margin") > 0.0


def test_shape_and_finite_on_real_board() -> None:
    board = [_fruit(0, 60.0, 500.0), _fruit(3, 200.0, 480.0), _fruit(5, 320.0, 460.0)]
    vec = board_features(board, sign=-1)
    assert vec.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(vec))
    assert _value(vec, "fruit_count") > 0.0
    assert _value(vec, "sign") == -1.0


def test_terms_are_separate_not_summed() -> None:
    """項ごとに別の列で出る。合計だけ渡すと学習側が重みを学び直せない。"""
    assert "size_order_pair" in FEATURE_NAMES
    assert "size_order_ideal" in FEATURE_NAMES
    # board_penalties の総和そのものを渡す列は持たない。
    assert not any("board_penalt" in name for name in FEATURE_NAMES)


def test_size_order_parts_sum_back_to_the_rule() -> None:
    """割った 2 項の和が本家の `_size_order_penalty` に戻る。

    ずれたまま集めると、学習側は存在しない量を見ることになる。
    """
    board = [_fruit(0, 40.0, 520.0), _fruit(6, 120.0, 460.0), _fruit(2, 300.0, 500.0)]
    vec = board_features(board, sign=1)
    total = _value(vec, "size_order_pair") + _value(vec, "size_order_ideal")
    assert abs(total - pen._size_order_penalty(board, 1)) < 1e-4


def test_corner_geometry_separates_lifted_from_seated() -> None:
    """角に着いた大実と、浮いた大実が別の値になる。

    `_corner_lift_penalty` は減点としては帯の外へ出なかった（NOTES「効かなかった
    案」）が、盤の性質としては角スイカの分かれ目そのものなので特徴には残す。
    """
    big_r = fruit_radius(MAX_FRUIT_TYPE)
    seated = [_fruit(MAX_FRUIT_TYPE, big_r, NORMALIZED_HEIGHT - big_r)]
    lifted = [_fruit(MAX_FRUIT_TYPE, big_r, NORMALIZED_HEIGHT - big_r - 80.0)]

    seated_vec = board_features(seated, sign=1)
    lifted_vec = board_features(lifted, sign=1)

    assert _value(seated_vec, "big_cornered") == 1.0
    assert _value(lifted_vec, "big_cornered") == 0.0
    assert _value(seated_vec, "big_floor_gap") < _value(lifted_vec, "big_floor_gap")


def test_watermelon_and_melon_are_counted_apart() -> None:
    board = [_fruit(MAX_FRUIT_TYPE, 100.0, 500.0), _fruit(MAX_FRUIT_TYPE - 1, 300.0, 500.0)]
    vec = board_features(board, sign=1)
    assert _value(vec, "watermelon_count") == 1.0
    assert _value(vec, "melon_count") == 1.0


def test_units_are_conserved_by_a_merge() -> None:
    """材料の総量は合体で保存される（NOTES「材料の計算」）。"""
    pair = [_fruit(3, 100.0, 500.0), _fruit(3, 200.0, 500.0)]
    merged = [_fruit(4, 150.0, 500.0)]
    assert abs(
        _value(board_features(pair, sign=1), "units")
        - _value(board_features(merged, sign=1), "units")
    ) < 1e-6
