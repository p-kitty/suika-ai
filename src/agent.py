"""離散列を選ぶ線形方策。numpy のみ。"""

from __future__ import annotations

import numpy as np

from .encode import OBS_DIM, encode
from .observe import Observation, clamp_drop_x
from .vision.normalized import NORMALIZED_WIDTH

# 落とす列のビン数。
N_ACTIONS = 20


def action_to_x(action: int, held_type: int | None) -> float:
    """離散行動 -> 正規化列。"""
    lo = 0.0
    hi = float(NORMALIZED_WIDTH)
    # ビン中央。
    x = (action + 0.5) * (hi - lo) / N_ACTIONS
    return clamp_drop_x(x, held_type)


class LinearPolicy:
    """softmax(W @ obs + b) の離散方策。"""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()
        scale = 0.01
        self.weight = self.rng.normal(0.0, scale, size=(N_ACTIONS, OBS_DIM)).astype(
            np.float64
        )
        self.bias = np.zeros(N_ACTIONS, dtype=np.float64)

    def logits(self, obs_vec: np.ndarray) -> np.ndarray:
        return self.weight @ obs_vec.astype(np.float64) + self.bias

    def probs(self, obs_vec: np.ndarray) -> np.ndarray:
        z = self.logits(obs_vec)
        z = z - z.max()
        exp = np.exp(z)
        return exp / exp.sum()

    def act(
        self, obs: Observation, *, greedy: bool = False
    ) -> tuple[int, float, np.ndarray]:
        """(action, x, obs_vec) を返す。"""
        vec = encode(obs)
        probs = self.probs(vec)
        if greedy:
            action = int(probs.argmax())
        else:
            action = int(self.rng.choice(N_ACTIONS, p=probs))
        return action, action_to_x(action, obs.held_type), vec

    def update(
        self,
        batch_obs: list[np.ndarray],
        batch_actions: list[int],
        batch_advantages: list[float],
        *,
        lr: float = 0.01,
    ) -> float:
        """REINFORCE。平均損失 (負の期待報酬近似) を返す。"""
        if not batch_obs:
            return 0.0
        grad_w = np.zeros_like(self.weight)
        grad_b = np.zeros_like(self.bias)
        loss = 0.0
        for obs_vec, action, adv in zip(batch_obs, batch_actions, batch_advantages):
            probs = self.probs(obs_vec)
            loss -= adv * np.log(probs[action] + 1e-12)
            # d log π(a|s) / d logits = one_hot - probs
            dlog = -probs
            dlog[action] += 1.0
            grad_w += adv * np.outer(dlog, obs_vec)
            grad_b += adv * dlog
        n = float(len(batch_obs))
        self.weight += lr * grad_w / n
        self.bias += lr * grad_b / n
        return float(loss / n)
