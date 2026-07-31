"""The reward for learning. Only merge points identical to the real Suika Game."""

from __future__ import annotations

from collections.abc import Sequence

from .observe import Observation
from .vision.colors import MAX_FRUIT_TYPE

WATERMELON = MAX_FRUIT_TYPE
# Losing when the crown rises above this y (y points down).
GAME_OVER_Y = 40.0

# Points when a fruit of that stage is made by a merge (index = type made).
# Cherries are only dropped and never made by merging, so 0.
# A double watermelon clear is CLEAR_SCORE, outside CREATE_SCORE.
CREATE_SCORE: tuple[int, ...] = (
    0,   # cherry
    1,   # straw
    3,   # grape
    6,   # dekopon
    10,  # orange
    15,  # apple
    21,  # pear
    28,  # peach
    36,  # pineapple
    45,  # melon
    55,  # watermelon
)
# When watermelons merge with each other and disappear.
CLEAR_SCORE = 65


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


def merge_points(source_type: int) -> int:
    """The real-game score when merging two fruits of source_type."""
    if source_type >= WATERMELON:
        return CLEAR_SCORE
    created = source_type + 1
    if 0 <= created < len(CREATE_SCORE):
        return CREATE_SCORE[created]
    return 0


def step_reward(
    before: Observation,
    after: Observation,
    *,
    merges: int = 0,
    merge_types: Sequence[int] = (),
    done: bool = False,
    win: bool = False,
) -> float:
    """The reward for one move = the sum of merge points of that move. No penalties or survival bonus.

    before/after/merges/done/win are for the caller's end-of-game checks and not used for points.
    """
    _ = (before, after, merges, done, win)
    return float(sum(merge_points(t) for t in merge_types))
