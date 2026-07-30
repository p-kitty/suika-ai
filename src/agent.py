"""離散列を選ぶ方策。numpy のみ。"""

from __future__ import annotations

from os import PathLike

import numpy as np

from .encode import OBS_DIM, encode
from .observe import Observation, clamp_drop_x
from .vision.normalized import NORMALIZED_WIDTH

# 落とす列のビン数。
N_ACTIONS = 20
HIDDEN = 64


def action_to_x(action: int, held_type: int | None) -> float:
    """離散行動 -> 正規化列。"""
    lo = 0.0
    hi = float(NORMALIZED_WIDTH)
    # ビン中央。
    x = (action + 0.5) * (hi - lo) / N_ACTIONS
    return clamp_drop_x(x, held_type)


def x_to_action(x: float) -> int:
    """正規化列 -> 最も近い離散行動。"""
    width = float(NORMALIZED_WIDTH)
    if width <= 0:
        return 0
    action = int(x / width * N_ACTIONS)
    return max(0, min(N_ACTIONS - 1, action))


class LinearPolicy:
    """1 隠れ層 MLP + softmax。名前は互換のため残す。"""

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        *,
        hidden: int = HIDDEN,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.hidden = hidden
        scale = 0.05
        self.w1 = self.rng.normal(0.0, scale, size=(hidden, OBS_DIM)).astype(np.float64)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = self.rng.normal(0.0, scale, size=(N_ACTIONS, hidden)).astype(
            np.float64
        )
        self.b2 = np.zeros(N_ACTIONS, dtype=np.float64)

    def _forward(self, obs_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = obs_vec.astype(np.float64)
        pre = self.w1 @ x + self.b1
        h = np.maximum(pre, 0.0)
        logits = self.w2 @ h + self.b2
        return h, logits

    def logits(self, obs_vec: np.ndarray) -> np.ndarray:
        return self._forward(obs_vec)[1]

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
        """REINFORCE / BC 共通。平均損失 (負の加重対数尤度) を返す。"""
        if not batch_obs:
            return 0.0
        gw1 = np.zeros_like(self.w1)
        gb1 = np.zeros_like(self.b1)
        gw2 = np.zeros_like(self.w2)
        gb2 = np.zeros_like(self.b2)
        loss = 0.0
        for obs_vec, action, adv in zip(batch_obs, batch_actions, batch_advantages):
            x = obs_vec.astype(np.float64)
            h, logits = self._forward(x)
            z = logits - logits.max()
            exp = np.exp(z)
            probs = exp / exp.sum()
            loss -= adv * np.log(probs[action] + 1e-12)

            dlog = -probs
            dlog[action] += 1.0
            dlogits = adv * dlog

            gw2 += np.outer(dlogits, h)
            gb2 += dlogits
            dh = self.w2.T @ dlogits
            dh *= (h > 0.0).astype(np.float64)
            gw1 += np.outer(dh, x)
            gb1 += dh

        n = float(len(batch_obs))
        self.w1 += lr * gw1 / n
        self.b1 += lr * gb1 / n
        self.w2 += lr * gw2 / n
        self.b2 += lr * gb2 / n
        return float(loss / n)

    def bc_update(
        self,
        batch_obs: list[np.ndarray],
        batch_actions: list[int],
        *,
        lr: float = 0.05,
    ) -> float:
        """教師行動へのクロスエントロピー。平均 NLL を返す。"""
        return self.update(
            batch_obs, batch_actions, [1.0] * len(batch_actions), lr=lr
        )

    def save(self, path: str | PathLike) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load(self, path: str | PathLike) -> None:
        data = np.load(path)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.hidden = int(self.w1.shape[0])
