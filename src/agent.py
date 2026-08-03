"""A policy that picks a discrete column. numpy only."""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike

import numpy as np

from .encode import OBS_DIM, encode
from .observe import Observation, clamp_drop_x
from .vision.normalized import NORMALIZED_WIDTH

# Number of bins for the drop column (the finer, the closer it can get to the teacher's continuous x).
N_ACTIONS = 32
HIDDEN = 128


def action_to_x(action: int, held_type: int | None) -> float:
    """Discrete action -> normalized column."""
    # Bin center.
    x = (action + 0.5) * float(NORMALIZED_WIDTH) / N_ACTIONS
    return clamp_drop_x(x, held_type)


def x_to_action(x: float) -> int:
    """Normalized column -> the nearest discrete action."""
    width = float(NORMALIZED_WIDTH)
    if width <= 0:
        return 0
    action = int(x / width * N_ACTIONS)
    return max(0, min(N_ACTIONS - 1, action))


def teacher_action_target(x: float) -> np.ndarray:
    """A distribution splitting the teacher's continuous x softly over nearby bins."""
    width = float(NORMALIZED_WIDTH)
    centers = (np.arange(N_ACTIONS, dtype=np.float64) + 0.5) * width / N_ACTIONS
    sigma = max(width / N_ACTIONS, 1e-6)
    z = -0.5 * ((centers - float(x)) / sigma) ** 2
    z = z - z.max()
    exp = np.exp(z)
    return exp / exp.sum()


class LinearPolicy:
    """1-hidden-layer MLP + softmax. The name is kept for compatibility."""

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
        """Return (action, x, obs_vec)."""
        vec = encode(obs)
        probs = self.probs(vec)
        if greedy:
            action = int(probs.argmax())
        else:
            action = int(self.rng.choice(N_ACTIONS, p=probs))
        return action, action_to_x(action, obs.held_type), vec

    def _fit(
        self,
        batch_obs: list[np.ndarray],
        gradient: Callable[[np.ndarray, int], tuple[np.ndarray, float]],
        *,
        lr: float,
    ) -> float:
        """Backpropagation and update for one batch. Returns the mean loss.

        Only the output-layer gradient differs, so it is delegated to gradient(probs, i) -> (dlogits, loss),
        and forward pass, accumulation, averaging and applying are shared between REINFORCE and BC.
        """
        if not batch_obs:
            return 0.0
        gw1 = np.zeros_like(self.w1)
        gb1 = np.zeros_like(self.b1)
        gw2 = np.zeros_like(self.w2)
        gb2 = np.zeros_like(self.b2)
        loss = 0.0
        for i, obs_vec in enumerate(batch_obs):
            x = obs_vec.astype(np.float64)
            h, logits = self._forward(x)
            z = logits - logits.max()
            exp = np.exp(z)
            probs = exp / exp.sum()

            dlogits, sample_loss = gradient(probs, i)
            loss += sample_loss

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

    def update(
        self,
        batch_obs: list[np.ndarray],
        batch_actions: list[int],
        batch_advantages: list[float],
        *,
        lr: float = 0.01,
        entropy_coef: float = 0.0,
    ) -> float:
        """Shared by REINFORCE / BC. Returns the mean loss (negative weighted log-likelihood)."""

        def gradient(probs: np.ndarray, i: int) -> tuple[np.ndarray, float]:
            action = batch_actions[i]
            adv = batch_advantages[i]
            loss = -adv * float(np.log(probs[action] + 1e-12))

            dlog = -probs
            dlog[action] += 1.0
            dlogits = adv * dlog
            if entropy_coef > 0.0:
                log_p = np.log(probs + 1e-12)
                entropy = -float(np.sum(probs * log_p))
                loss -= entropy_coef * entropy
                dlogits += entropy_coef * probs * (log_p + entropy)
            return dlogits, loss

        return self._fit(batch_obs, gradient, lr=lr)

    def bc_update(
        self,
        batch_obs: list[np.ndarray],
        batch_actions: list[int],
        *,
        lr: float = 0.05,
    ) -> float:
        """Cross-entropy to the teacher action. Returns the mean NLL."""
        return self.update(
            batch_obs, batch_actions, [1.0] * len(batch_actions), lr=lr
        )

    def bc_update_dist(
        self,
        batch_obs: list[np.ndarray],
        batch_targets: list[np.ndarray],
        *,
        lr: float = 0.05,
    ) -> float:
        """Cross-entropy to a soft teacher distribution."""

        def gradient(probs: np.ndarray, i: int) -> tuple[np.ndarray, float]:
            t = batch_targets[i].astype(np.float64)
            loss = -float(np.sum(t * np.log(probs + 1e-12)))
            return t - probs, loss

        return self._fit(batch_obs, gradient, lr=lr)

    def snapshot(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.w1.copy(),
            "b1": self.b1.copy(),
            "w2": self.w2.copy(),
            "b2": self.b2.copy(),
        }

    def restore(self, data: dict[str, np.ndarray]) -> None:
        self.w1 = data["w1"].copy()
        self.b1 = data["b1"].copy()
        self.w2 = data["w2"].copy()
        self.b2 = data["b2"].copy()
        self.hidden = int(self.w1.shape[0])

    def save(self, path: str | PathLike) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load(self, path: str | PathLike) -> None:
        data = np.load(path)
        w1, w2 = data["w1"], data["w2"]
        if w1.shape[1] != OBS_DIM or w2.shape[0] != N_ACTIONS:
            raise ValueError(
                f"checkpoint shape mismatch: got w1={w1.shape} w2={w2.shape}, "
                f"expected (*, {OBS_DIM}) and ({N_ACTIONS}, *). Re-train."
            )
        self.restore(
            {
                "w1": w1,
                "b1": data["b1"],
                "w2": w2,
                "b2": data["b2"],
            }
        )
