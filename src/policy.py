"""Decide the drop column. Heuristics for now."""

from __future__ import annotations

import math

from .observe import Observation, clamp_drop_x
from .vision.classify import fruit_radius_ratios
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# Spacing of candidate columns (normalized coordinates).
CANDIDATE_STEP = 8.0
# Tolerance for a contact that could merge (difference between center distance and sum of radii).
MERGE_SLACK = 18.0
# Dangerous if the head rises above this y (near the top edge of the board).
DANGER_Y = 90.0


def choose_x(obs: Observation) -> float:
    """Return the column to drop from the observation. Assumes ready with held_type present."""
    if obs.held_type is None:
        raise ValueError("no held_type")

    held_r = _radius(obs.held_type)
    best_x = NORMALIZED_WIDTH / 2
    best_score = -math.inf

    for x in _candidates(obs, held_r):
        x = clamp_drop_x(x, obs.held_type)
        score = _score(obs, x, held_r)
        if score > best_score:
            best_score = score
            best_x = x

    return best_x


def _candidates(obs: Observation, held_r: float) -> list[float]:
    """Uniform spacing plus spots above / beside same types."""
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    xs = {round(x / CANDIDATE_STEP) * CANDIDATE_STEP for x in _frange(lo, hi, CANDIDATE_STEP)}

    for fruit in obs.fruits:
        if fruit.type != obs.held_type:
            continue
        xs.add(fruit.x)
        # Aim to merge by placing beside it.
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)

    if obs.next_type is not None:
        for fruit in obs.fruits:
            if fruit.type != obs.next_type:
                continue
            xs.add(fruit.x)

    return [x for x in xs if lo <= x <= hi]


def _score(obs: Observation, x: float, held_r: float) -> float:
    land_y = _land_y(obs.fruits, x, held_r)
    score = 0.0

    # The lower it falls, the better (larger y).
    score += land_y

    # A big bonus if it is likely to touch a same type.
    score += 120.0 * _merge_chance(obs.fruits, obs.held_type, x, land_y, held_r)

    # Slightly favor being near a same type as the next fruit, without breaking the board too much.
    if obs.next_type is not None:
        score += 25.0 * _merge_chance(obs.fruits, obs.next_type, x, land_y, held_r)

    # Penalty as the head approaches the top edge.
    crown = land_y - held_r
    if crown < DANGER_Y:
        score -= (DANGER_Y - crown) * 3.0

    # Slightly favor dropping into an empty column (early scattering).
    if not _column_fruits(obs.fruits, x, held_r):
        score += 8.0

    # Center on ties. So an empty board does not choose only the edges.
    score -= abs(x - NORMALIZED_WIDTH / 2) * 0.05

    return score


def _land_y(fruits: tuple[Fruit, ...] | list[Fruit], x: float, held_r: float) -> float:
    """Center y when dropped at column x. The floor, or on top of the crown of an overlapping fruit."""
    top = float(NORMALIZED_HEIGHT)
    for fruit in fruits:
        if abs(fruit.x - x) > fruit.radius + held_r:
            continue
        top = min(top, fruit.y - fruit.radius)
    return top - held_r


def _merge_chance(
    fruits: tuple[Fruit, ...] | list[Fruit],
    fruit_type: int | None,
    x: float,
    land_y: float,
    held_r: float,
) -> float:
    """0-1. High if the landing is likely to touch a same type."""
    if fruit_type is None:
        return 0.0

    best = 0.0
    for fruit in fruits:
        if fruit.type != fruit_type:
            continue
        dist = math.hypot(x - fruit.x, land_y - fruit.y)
        expected = held_r + fruit.radius
        gap = abs(dist - expected)
        if gap >= MERGE_SLACK:
            # Dropping directly on top also tends to be a merge target.
            if abs(x - fruit.x) <= max(held_r, fruit.radius) * 0.85:
                best = max(best, 0.55)
            continue
        best = max(best, 1.0 - gap / MERGE_SLACK)
    return best


def _column_fruits(
    fruits: tuple[Fruit, ...] | list[Fruit],
    x: float,
    held_r: float,
) -> list[Fruit]:
    return [f for f in fruits if abs(f.x - x) <= f.radius + held_r]


def _radius(fruit_type: int) -> float:
    return fruit_radius_ratios()[fruit_type] * NORMALIZED_WIDTH


def _frange(start: float, stop: float, step: float):
    x = start
    while x <= stop + 1e-6:
        yield x
        x += step
