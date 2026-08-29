"""Features of the post-drop board.

What is checked is **that the terms come out separately** and that the corner watermelon geometry is picked up.
Values themselves move with the weights in `penalties.py`, so they are not pinned.
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
    # An empty board is farthest from death. At 0 it could not be told from 'exactly on the losing line'.
    assert _value(vec, "crown_margin") > 0.0


def test_shape_and_finite_on_real_board() -> None:
    board = [_fruit(0, 60.0, 500.0), _fruit(3, 200.0, 480.0), _fruit(5, 320.0, 460.0)]
    vec = board_features(board, sign=-1)
    assert vec.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(vec))
    assert _value(vec, "fruit_count") > 0.0
    assert _value(vec, "sign") == -1.0


def test_terms_are_separate_not_summed() -> None:
    """Each term comes out in its own column. Passing only the sum leaves the learner unable to relearn weights."""
    assert "size_order_pair" in FEATURE_NAMES
    assert "size_order_ideal" in FEATURE_NAMES
    # There is no column passing the sum of board_penalties itself.
    assert not any("board_penalt" in name for name in FEATURE_NAMES)


def test_size_order_parts_sum_back_to_the_rule() -> None:
    """The two split terms add back up to the original `_size_order_penalty`.

    Collecting while they are off would make the learner look at a quantity that does not exist.
    """
    board = [_fruit(0, 40.0, 520.0), _fruit(6, 120.0, 460.0), _fruit(2, 300.0, 500.0)]
    vec = board_features(board, sign=1)
    total = _value(vec, "size_order_pair") + _value(vec, "size_order_ideal")
    assert abs(total - pen._size_order_penalty(board, 1)) < 1e-4


def test_corner_geometry_separates_lifted_from_seated() -> None:
    """A big fruit in the corner and a lifted big fruit get different values.

    `_corner_lift_penalty` did not escape the band as a penalty (NOTES 'Ideas that did not
    work'), but as a board property it is the corner watermelon dividing line itself, so it stays as a feature.
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
    """The total material is conserved by merges (NOTES 'Material arithmetic')."""
    pair = [_fruit(3, 100.0, 500.0), _fruit(3, 200.0, 500.0)]
    merged = [_fruit(4, 150.0, 500.0)]
    assert abs(
        _value(board_features(pair, sign=1), "units")
        - _value(board_features(merged, sign=1), "units")
    ) < 1e-6
