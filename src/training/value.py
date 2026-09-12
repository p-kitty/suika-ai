"""Read and write the learned value function V, and turn one board into a value.

Training itself is `scripts/train_value.py` (ridge, closed form, numpy only). This is
**the place that makes the fitted coefficients usable from `choose_x`**; it holds no model.
It is separate because the policy side should not import a training script.

`sign` passed to `board_features` is the direction of the board **before the drop** (`policy._order_sign`).
The collection side (`training/collect.py`) builds features with the pre-drop sign, so at inference
recomputing it from the post-drop board would disagree with training. That is why the caller passes it.

Weights apply to standardized features. `mean` / `std` come from the train split used for fitting,
and unless saved together with the model the same values cannot be reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import numpy as np

from .features import FEATURE_DIM, board_features
from ..vision.state import Fruit


@dataclass(frozen=True)
class LinearValue:
    """Standardization + linear V. Uses only the columns listed in `use`.

    `bias` is constant among candidates of the same position so it does not affect ranking, but it is needed
    when looking at the value itself (calibrating λ), so it is kept.
    """

    mean: np.ndarray  # (FEATURE_DIM,)
    std: np.ndarray  # (FEATURE_DIM,)
    use: np.ndarray  # (K,) column indices of the features used
    coef: np.ndarray  # (K,)
    bias: float

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """(N, FEATURE_DIM) feature matrix -> (N,) V."""
        z = (feats - self.mean) / self.std
        return z[:, self.use] @ self.coef + self.bias

    def boards(
        self, boards: list[list[Fruit]] | list[tuple[Fruit, ...]], *, sign: int
    ) -> np.ndarray:
        """Return V for a list of post-drop boards. sign is the direction of the board **before** the drop."""
        if not boards:
            return np.zeros(0, dtype=np.float64)
        feats = np.empty((len(boards), FEATURE_DIM), dtype=np.float32)
        for i, board in enumerate(boards):
            feats[i] = board_features(board, sign=sign)
        return self.predict(feats)


def save(model: LinearValue, path: str | PathLike) -> None:
    np.savez(
        path,
        mean=model.mean,
        std=model.std,
        use=model.use,
        coef=model.coef,
        bias=np.float64(model.bias),
    )


def load(path: str | PathLike) -> LinearValue:
    data = np.load(path)
    return LinearValue(
        mean=data["mean"].astype(np.float64),
        std=data["std"].astype(np.float64),
        use=data["use"].astype(np.intp),
        coef=data["coef"].astype(np.float64),
        bias=float(data["bias"]),
    )
