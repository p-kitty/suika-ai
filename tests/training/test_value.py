"""Reading and writing the learned V, and the path that turns one board into a value.

The fitting itself (`scripts/train_value.py`) is not checked. What is checked is
**that saving and loading gives the same values** and **that the caller can pass sign**.
"""

import numpy as np

from src.training.features import FEATURE_DIM, FEATURE_NAMES, board_features
from src.training.value import LinearValue, load, save
from src.vision.classify import fruit_radius
from src.vision.state import Fruit


def _fruit(fruit_type: int, x: float, y: float) -> Fruit:
    return Fruit(
        type=fruit_type, x=x, y=y, radius=fruit_radius(fruit_type), confidence=90
    )


def _model(name: str, coef: float, *, bias: float = 0.0) -> LinearValue:
    """An artificial model looking at a single feature. Standardization is the identity."""
    return LinearValue(
        mean=np.zeros(FEATURE_DIM),
        std=np.ones(FEATURE_DIM),
        use=np.array([FEATURE_NAMES.index(name)]),
        coef=np.array([coef]),
        bias=bias,
    )


def test_predict_is_the_standardized_linear_form() -> None:
    model = LinearValue(
        mean=np.full(FEATURE_DIM, 2.0),
        std=np.full(FEATURE_DIM, 4.0),
        use=np.array([0, 3]),
        coef=np.array([10.0, -1.0]),
        bias=0.5,
    )
    feats = np.zeros((1, FEATURE_DIM))
    feats[0, 0] = 6.0
    feats[0, 3] = 10.0
    # z = (6-2)/4 = 1.0, (10-2)/4 = 2.0 -> 10*1 - 1*2 + 0.5
    assert model.predict(feats)[0] == 8.5


def test_save_load_roundtrip(tmp_path) -> None:
    model = _model("fruit_count", 3.0, bias=-1.5)
    path = tmp_path / "v.npz"
    save(model, path)
    back = load(path)

    feats = np.zeros((2, FEATURE_DIM))
    feats[0, FEATURE_NAMES.index("fruit_count")] = 1.0
    feats[1, FEATURE_NAMES.index("fruit_count")] = 2.0
    assert np.allclose(model.predict(feats), back.predict(feats))


def test_boards_matches_board_features() -> None:
    model = _model("fruit_count", 5.0)
    boards = [
        [_fruit(0, 60.0, 500.0)],
        [_fruit(0, 60.0, 500.0), _fruit(1, 200.0, 490.0)],
    ]
    got = model.boards(boards, sign=1)
    want = model.predict(np.stack([board_features(b, sign=1) for b in boards]))
    assert np.allclose(got, want)
    # More fruits means a larger fruit_count = a different value per board.
    assert got[1] > got[0]


def test_sign_comes_from_the_caller() -> None:
    """Sign-dependent terms follow the direction the caller passed (not recomputed from the post-drop board)."""
    model = _model("sign", 1.0)
    board = [_fruit(5, 60.0, 470.0), _fruit(0, 300.0, 500.0)]
    assert model.boards([board], sign=1)[0] == 1.0
    assert model.boards([board], sign=-1)[0] == -1.0


def test_empty_board_list_is_empty() -> None:
    assert len(_model("fruit_count", 1.0).boards([], sign=1)) == 0
