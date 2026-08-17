"""観測を固定長ベクトルにする。学習入力用。"""

from __future__ import annotations

import numpy as np

from ..observe import Observation
from ..vision.colors import FRUIT_NAMES
from ..vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH

N_TYPES = len(FRUIT_NAMES)
# 大きい実を優先して載せる上限。中盤以降 20 個を超える盤も珍しくなく
# (実測: 終盤 23 個)、旧 16 だと真っ先に切り捨てられるのが cherry/strawberry
# など低段位の散在実 — まさに盤面を埋め尽くして事故を招く張本人が学習側から
# 見えていなかった。32 に上げて実質切り捨てなしにする。
MAX_FRUITS = 32
# 各実: type_norm, x_norm, y_norm, r_norm
FRUIT_DIM = 4
# held one-hot + next one-hot + fruit slots
OBS_DIM = N_TYPES * 2 + MAX_FRUITS * FRUIT_DIM


def encode(obs: Observation) -> np.ndarray:
    """Observation -> float32 ベクトル (OBS_DIM,)。"""
    out = np.zeros(OBS_DIM, dtype=np.float32)
    if obs.held_type is not None:
        out[obs.held_type] = 1.0
    if obs.next_type is not None:
        out[N_TYPES + obs.next_type] = 1.0

    fruits = sorted(obs.fruits, key=lambda f: (-f.type, f.y, f.x))[:MAX_FRUITS]
    base = N_TYPES * 2
    for i, fruit in enumerate(fruits):
        o = base + i * FRUIT_DIM
        out[o] = fruit.type / max(N_TYPES - 1, 1)
        out[o + 1] = fruit.x / NORMALIZED_WIDTH
        out[o + 2] = fruit.y / NORMALIZED_HEIGHT
        out[o + 3] = fruit.radius / NORMALIZED_WIDTH
    return out
