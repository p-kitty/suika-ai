"""Turn an observation into a fixed-length vector. For training input."""

from __future__ import annotations

import numpy as np

from .observe import Observation
from .vision.colors import FRUIT_NAMES
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH

N_TYPES = len(FRUIT_NAMES)
# Cap on fruits loaded, biggest first.
MAX_FRUITS = 16
# Per fruit: type_norm, x_norm, y_norm, r_norm
FRUIT_DIM = 4
# held one-hot + next one-hot + fruit slots
OBS_DIM = N_TYPES * 2 + MAX_FRUITS * FRUIT_DIM


def encode(obs: Observation) -> np.ndarray:
    """Observation -> float32 vector (OBS_DIM,)."""
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
