"""Turn a post-drop board into fixed-length features. Input for the value function.

A different role from `encode.py`. That one passes the board **before the drop** as a list of fruits,
with no per-candidate landing results. A 1-hidden-layer MLP imitating the teacher's judgment from it
plateaued at match 30% even with an exhaustive sweep of lr / epoch / hidden / soft-hard
(NOTES 'Investigated: BC does not reach 60-70% match'). This one counts the board after actually dropping
each candidate, to give the learner the same information the teacher has.

**Pass each term separately. Do not sum.** Passing the sum of `board_penalties` or `eval`
as a single number leaves the learner unable to relearn the weighting. Weights staying applied
is fine (absorbed by the linear coefficients), but **once summed they cannot be separated**.
`size_order` is the sum of two rules of different nature, so it is taken as the two columns
`penalties.size_order_parts` returns (NOTES 'Split composite terms into sub-terms').

Quantities that matter for the corner watermelon (the target shape) are on the geometry side. The wall and floor gaps of the biggest fruit are
exactly the dividing line measured over 20 traced games in NOTES 'Investigated: why corner watermelons do not happen',
and are known to be inexpressible as penalties in `penalties.py`
(`_corner_lift_penalty` escapes the band 0.0%. →NOTES 'Ideas that did not work').
"""

from __future__ import annotations

import numpy as np

from .. import penalties as pen
from ..reward import CORNER_SLACK, GAME_OVER_Y, WATERMELON
from ..vision.colors import MAX_FRUIT_TYPE
from ..vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from ..vision.state import Fruit

# Feature order. Named for debugging and for reading weights downstream.
FEATURE_NAMES: tuple[str, ...] = (
    "fruit_count",
    "crown_margin",
    "max_type",
    "units",
    "bury_pair",
    "bury_lone",
    "perch",
    "pit",
    "excess_same",
    "size_order_pair",
    "size_order_ideal",
    "corner_pocket",
    "big_wall_gap",
    "big_floor_gap",
    "big_cornered",
    "melon_count",
    "watermelon_count",
    "mean_height",
    "sign",
)
FEATURE_DIM = len(FEATURE_NAMES)

# Denominators. Just rough scales to keep features near 0-1; they mean nothing.
_COUNT_SCALE = 32.0
_UNIT_SCALE = 1024.0


def board_features(fruits: list[Fruit] | tuple[Fruit, ...], *, sign: int) -> np.ndarray:
    """Post-drop board -> float32 vector (FEATURE_DIM,).

    sign is the size direction of the board (`policy._order_sign`). Corner pocket, size order and
    wall gaps change value depending on which side is taken as big, so the caller passes it.
    """
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    board = list(fruits)
    if not board:
        # Empty board. Only crown_margin is set to 1.0, meaning 'farthest from death'.
        out[FEATURE_NAMES.index("crown_margin")] = 1.0
        out[FEATURE_NAMES.index("sign")] = float(sign)
        return out

    max_t = max(f.type for f in board)
    biggest = max(board, key=lambda f: (f.type, -f.y))
    crown = min(f.y - f.radius for f in board)
    pair, lone = pen._bury_counts(board)
    so_pair, so_ideal = pen.size_order_parts(board, sign)

    values = {
        "fruit_count": len(board) / _COUNT_SCALE,
        # Margin to the losing line. Negative means dead.
        "crown_margin": (crown - GAME_OVER_Y) / NORMALIZED_HEIGHT,
        "max_type": max_t / MAX_FRUIT_TYPE,
        # A quantity conserved by merges = the total material stacked on the board (NOTES 'Material arithmetic').
        "units": sum(2.0**f.type for f in board) / _UNIT_SCALE,
        "bury_pair": pair,
        "bury_lone": lone,
        "perch": pen._perch_penalty(board),
        "pit": pen._pit_penalty(board),
        "excess_same": pen._excess_same_penalty(board),
        "size_order_pair": so_pair,
        "size_order_ideal": so_ideal,
        "corner_pocket": pen._corner_pocket_penalty(board, sign),
        # The corner watermelon dividing line. The wall side looks only at the big side (`pen.wall_gap` depends on sign).
        "big_wall_gap": pen.wall_gap(biggest, sign) / NORMALIZED_WIDTH,
        "big_floor_gap": (NORMALIZED_HEIGHT - (biggest.y + biggest.radius))
        / NORMALIZED_HEIGHT,
        "big_cornered": 0.0,
        "melon_count": sum(1 for f in board if f.type == WATERMELON - 1),
        "watermelon_count": sum(1 for f in board if f.type == WATERMELON),
        "mean_height": sum(NORMALIZED_HEIGHT - f.y for f in board)
        / len(board)
        / NORMALIZED_HEIGHT,
        "sign": float(sign),
    }
    # Whether it touches both wall and floor. Uses the same tolerance as the corner watermelon check (`is_corner_watermelon`).
    # Any type (a corner at the melon stage also shows as progress).
    values["big_cornered"] = float(
        pen.wall_gap(biggest, sign) <= CORNER_SLACK
        and NORMALIZED_HEIGHT - (biggest.y + biggest.radius) <= CORNER_SLACK
    )

    for i, name in enumerate(FEATURE_NAMES):
        out[i] = values[name]
    return out
