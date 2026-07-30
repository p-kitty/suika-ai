"""The reward for learning. The goal is a successful end by clearing a double watermelon."""

from __future__ import annotations

from .observe import Observation
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT

WATERMELON = MAX_FRUIT_TYPE
# Losing when the crown rises above this y (y points down).
GAME_OVER_Y = 40.0

# Surviving one move.
STEP_REWARD = 0.05
# Per merge (heavier for bigger fruits).
MERGE_WEIGHT = 1.0
# When the max type on the board grows.
PROGRESS_WEIGHT = 2.0
# When a watermelon is newly added.
WATERMELON_BONUS = 20.0
# The moment there are 2 or more watermelons (not a keeping bonus).
DOUBLE_REACH_BONUS = 15.0
# When watermelons decrease / disappear through a merge.
WATERMELON_CLEAR_BONUS = 25.0
# A successful end by a double watermelon clear (the highest).
WIN_BONUS = 200.0
# Backward-compatible aliases (old names, reaching bonus).
DOUBLE_WATERMELON_BONUS = DOUBLE_REACH_BONUS
# Game over.
DEATH_PENALTY = -20.0


def is_game_over(obs: Observation) -> bool:
    """Whether the crown is past the losing line."""
    if not obs.fruits:
        return False
    crown = min(f.y - f.radius for f in obs.fruits)
    return crown < GAME_OVER_Y


def watermelon_count(obs: Observation) -> int:
    return sum(1 for f in obs.fruits if f.type == WATERMELON)


def cleared_double_watermelon(
    before: Observation,
    after: Observation,
    *,
    merges: int,
) -> bool:
    """Whether two or more watermelons decreased through a merge (the success condition)."""
    if merges <= 0:
        return False
    before_w = watermelon_count(before)
    after_w = watermelon_count(after)
    return before_w >= 2 and after_w < before_w


def _max_fruit_type(obs: Observation) -> int:
    if not obs.fruits:
        return -1
    return max(f.type for f in obs.fruits)


def step_reward(
    before: Observation,
    after: Observation,
    *,
    merges: int,
    done: bool,
    win: bool = False,
) -> float:
    """The reward for one move.

    - survival and merges are the base reward
    - bonuses for updating the board's max stage, more watermelons, reaching a double, clearing watermelons
    - WIN_BONUS (the highest) on a successful end by a double clear
    - a big penalty on game over
    """
    if done and not win:
        return DEATH_PENALTY

    reward = STEP_REWARD
    if merges > 0:
        # Roughly weight by the max type after merging (merges only assuming held if none).
        grown = _max_fruit_type(after)
        weight = MERGE_WEIGHT * (1.0 + max(grown, 0) * 0.15)
        reward += merges * weight

    before_max = _max_fruit_type(before)
    after_max = _max_fruit_type(after)
    if after_max > before_max:
        reward += (after_max - before_max) * PROGRESS_WEIGHT

    before_w = watermelon_count(before)
    after_w = watermelon_count(after)
    if after_w > before_w:
        reward += (after_w - before_w) * WATERMELON_BONUS
    if before_w < 2 <= after_w:
        reward += DOUBLE_REACH_BONUS
    if merges > 0 and after_w < before_w:
        reward += (before_w - after_w) * WATERMELON_CLEAR_BONUS

    if win:
        reward += WIN_BONUS

    # A tall pile approaches future death, so a small penalty (just before instant death).
    if after.fruits:
        crown = min(f.y - f.radius for f in after.fruits)
        headroom = (crown - GAME_OVER_Y) / max(NORMALIZED_HEIGHT - GAME_OVER_Y, 1.0)
        if headroom < 0.25:
            reward -= (0.25 - headroom) * 2.0

    return float(reward)
