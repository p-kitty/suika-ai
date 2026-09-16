"""One sim episode played by a given move chooser, and the metrics the evaluation scripts report.

`compare_policy.py`, `eval_policy.py` and `eval_ranker.py` each carried a copy of this loop, and the copies
drifted: eval_ranker read `max_type` / `max_wm` off the final board, which differs from the running maximum
exactly when a double watermelon clears, so its numbers could not be set beside compare_policy's.
Every script now gets every metric and prints the ones it wants.

Not a script. It has no sys.path insert; by the time a script imports it ROOT is already on the path
(see `_bootstrap.py`).
"""

from __future__ import annotations

from collections.abc import Callable

from src.observe import Observation
from src.reward import is_corner_watermelon, watermelon_count
from src.sim.sim_env import SimEnv

# Moves considered early game. How the board breaks up to here is looked at separately from later.
EARLY_STEPS = 30
# Merges per move counted as a cascade firing.
CASCADE_MERGES = 3


def play_episode(
    seed: int, choose: Callable[[Observation], float], max_steps: int
) -> dict[str, float]:
    """Play seed to a natural end or max_steps and return its metrics, all as floats (they go to JSON and stats)."""
    env = SimEnv(seed=seed)
    obs = env.reset()
    score = 0.0
    early_score = 0.0
    merges = 0
    cascades = 0
    steps = 0
    max_type = -1
    max_wm = 0
    corner_wm = False
    early_crowns: list[float] = []
    info = "ok"
    for _ in range(max_steps):
        result = env.step(choose(obs))
        obs = result.observation
        score += result.score
        merges += result.merges
        steps += 1
        info = result.info
        if result.merges >= CASCADE_MERGES:
            cascades += 1
        if steps <= EARLY_STEPS:
            early_score += result.score
            if obs.fruits:
                early_crowns.append(min(f.y - f.radius for f in obs.fruits))
        if obs.fruits:
            max_type = max(max_type, max(f.type for f in obs.fruits))
        max_wm = max(max_wm, watermelon_count(obs))
        corner_wm = corner_wm or is_corner_watermelon(obs.fruits)
        if result.done:
            break
    return {
        "seed": float(seed),
        "steps": float(steps),
        "score": score,
        "early_score": early_score,
        "merges": float(merges),
        "cascades": float(cascades),
        "max_type": float(max_type),
        # The most watermelons on the board at once. 2 means a double watermelon was lined up.
        "max_wm": float(max_wm),
        # Whether a corner watermelon (the target shape) was ever reached.
        "corner_wm": 1.0 if corner_wm else 0.0,
        # y points down. Smaller means a taller pile = dangerous.
        "early_crown": min(early_crowns) if early_crowns else float("nan"),
        # Whether it died before reaching the cap. A direct metric of early collapse.
        "dead_early": 1.0 if (info == "dead" and steps <= EARLY_STEPS) else 0.0,
        "dead": 1.0 if info == "dead" else 0.0,
        "win": 1.0 if info == "win" else 0.0,
        # Whether it was cut at max_steps without a natural end. Truncation is biased toward long games,
        # so a mean with them mixed in underestimates better policies (NOTES 'How to measure').
        "truncated": 0.0 if info in ("dead", "win") else 1.0,
    }
