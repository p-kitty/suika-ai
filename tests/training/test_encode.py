"""観測エンコードの単体テスト。"""

import numpy as np

from src.training.encode import N_TYPES, OBS_DIM, encode
from src.observe import Observation
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT
from src.vision.state import Fruit


def test_encode_shape_and_held_next() -> None:
    obs = Observation(
        ready=True,
        blocked=False,
        fruits=(),
        held_type=2,
        held_x=200.0,
        next_type=4,
    )
    vec = encode(obs)
    assert vec.shape == (OBS_DIM,)
    assert vec.dtype == np.float32
    assert vec[2] == 1.0
    assert vec[N_TYPES + 4] == 1.0


def test_encode_includes_fruits_largest_first() -> None:
    cherry_r = fruit_radius(0)
    apple_r = fruit_radius(5)
    fruits = (
        Fruit(type=0, x=50, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90),
        Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90),
    )
    obs = Observation(
        ready=True,
        blocked=False,
        fruits=fruits,
        held_type=0,
        held_x=100.0,
        next_type=1,
    )
    vec = encode(obs)
    # 最初のスロットは apple (type 5)。
    assert abs(vec[N_TYPES * 2] - 5 / (N_TYPES - 1)) < 1e-6
