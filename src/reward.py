"""The reward for learning. Only merge points identical to the real Suika Game."""

from __future__ import annotations

from collections.abc import Sequence

from .observe import Observation
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

WATERMELON = MAX_FRUIT_TYPE
# Losing when the crown rises above this y (y points down). Converted from the old
# basis of 40.0 by the amount the board moved to the inside-of-the-wall basis.
GAME_OVER_Y = 14.9

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


def is_lost(fruits: Sequence[Fruit]) -> bool:
    """Whether the crown is past the losing line. Takes the post-drop board as is.

    The policy uses it to sort candidates by 'does this move die',
    so it takes a list of fruits rather than an Observation (`policy.choose_x`).
    """
    if not fruits:
        return False
    crown = min(f.y - f.radius for f in fruits)
    return crown < GAME_OVER_Y


def is_game_over(obs: Observation) -> bool:
    return is_lost(obs.fruits)


# Tolerance for the corner check. A cherry (radius 14.2) fits as is into the corner gap left when a watermelon (radius 112)
# touches the wall and floor, and the watermelon does not move. A strawberry pushes
# the watermelon 4.9 from the wall, a grape 23.2 (both from the position of a circle touching the wall, floor and
# watermelon). Taking the middle, up to a strawberry counts as a corner.
CORNER_SLACK = 12.0


def is_corner_watermelon(fruits: Sequence[Fruit]) -> bool:
    """Whether there is a watermelon tight against the wall and floor.

    A corner watermelon is the target shape (→NOTES 'Current approach'). So that reaching it can be seen from outside,
    it is judged only from the board's fruits. Even with a small fruit wedged in the corner, if the amount pushed out
    fits within `CORNER_SLACK` it counts as a corner.
    """
    for fruit in fruits:
        if fruit.type != WATERMELON:
            continue
        if fruit.y + fruit.radius < NORMALIZED_HEIGHT - CORNER_SLACK:
            continue
        wall_gap = min(fruit.x - fruit.radius, NORMALIZED_WIDTH - fruit.x - fruit.radius)
        if wall_gap <= CORNER_SLACK:
            return True
    return False


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


def merge_score(merge_types: Sequence[int] = ()) -> float:
    """The real-game score of one move = the sum of merge points of that move. No penalties or survival bonus."""
    return float(sum(merge_points(t) for t in merge_types))
