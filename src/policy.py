"""Decide the drop column. Virtual drop + merge heuristics."""

from __future__ import annotations

import math
import statistics

from .observe import Observation, clamp_drop_x
from .vision.classify import fruit_radius_ratios
from .vision.colors import FRUIT_NAMES
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# Spacing of candidate columns (normalized coordinates).
CANDIDATE_STEP = 8.0
# Tolerance for a contact that could merge (difference between center distance and sum of radii). For candidate evaluation.
MERGE_SLACK = 18.0
# Contact for a virtual merge. The observed board is assumed still, so do not loosen too much.
CONTACT_SLACK = 2.0
# Dangerous if the head rises above this y (near the top edge of the board).
DANGER_Y = 90.0
# Column width for evaluating flatness.
FLAT_BIN = 40.0
# Watermelon. No merging beyond this.
MAX_FRUIT_TYPE = len(FRUIT_NAMES) - 1


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
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)

    if obs.next_type is not None:
        for fruit in obs.fruits:
            if fruit.type != obs.next_type:
                continue
            xs.add(fruit.x)
            gap = held_r + fruit.radius
            xs.add(fruit.x - gap)
            xs.add(fruit.x + gap)

    return [x for x in xs if lo <= x <= hi]


def _score(obs: Observation, x: float, held_r: float) -> float:
    """Score by looking at the board after the move."""
    before = list(obs.fruits)
    after, merges = _after_drop(obs, x)
    land_y = _land_y(before, x, held_r)

    score = 0.0

    score += 140.0 * merges
    if merges >= 2:
        score += 80.0 * (merges - 1)

    score += land_y * 0.35

    crown = _top_crown(after)
    score += crown * 0.8
    if crown < DANGER_Y:
        score -= (DANGER_Y - crown) * 4.0

    score -= 90.0 * _bury_penalty(after)

    if obs.next_type is not None:
        score += 70.0 * _next_setup_at(before, after, x, obs.next_type, merges)

    score -= 1.2 * _height_variance(after)

    if merges == 0 and not _column_fruits(before, x, held_r):
        score += 8.0

    score -= abs(x - NORMALIZED_WIDTH / 2) * 0.05

    return score


def _after_drop(obs: Observation, x: float) -> tuple[list[Fruit], int]:
    """Board and merge count after dropping at column x and resolving merges."""
    assert obs.held_type is not None
    fruits = list(obs.fruits)
    fruits, dropped = _place(fruits, obs.held_type, x)
    return _resolve_merges(fruits, active={dropped})


def _place(fruits: list[Fruit], fruit_type: int, x: float) -> tuple[list[Fruit], int]:
    """Land a fruit at column x and add it. Also returns the added index."""
    r = _radius(fruit_type)
    y = _land_y(fruits, x, r)
    fruits.append(Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=100.0))
    return fruits, len(fruits) - 1


def _resolve_merges(fruits: list[Fruit], active: set[int]) -> tuple[list[Fruit], int]:
    """Merge only same-type contacts starting from the dropped fruit. The observed board is assumed still."""
    fruits = list(fruits)
    merges = 0
    for _ in range(64):
        pair = _find_merge_pair(fruits, active)
        if pair is None:
            break
        i, j = pair
        a, b = fruits[i], fruits[j]
        new_type = a.type + 1
        mid_x = (a.x + b.x) / 2
        for idx in sorted((i, j), reverse=True):
            fruits.pop(idx)

        if new_type > MAX_FRUIT_TYPE:
            active = set()
            merges += 1
            continue

        fruits, new_i = _place(fruits, new_type, mid_x)
        active = {new_i}
        merges += 1

    return fruits, merges


def _find_merge_pair(fruits: list[Fruit], active: set[int]) -> tuple[int, int] | None:
    """Same-type pairs in contact with the active side."""
    for i in sorted(active):
        if i < 0 or i >= len(fruits):
            continue
        a = fruits[i]
        for j, b in enumerate(fruits):
            if j == i or b.type != a.type:
                continue
            if _touching(a, b):
                return (i, j) if i < j else (j, i)
    return None


def _touching(a: Fruit, b: Fruit) -> bool:
    dist = math.hypot(a.x - b.x, a.y - b.y)
    return dist <= a.radius + b.radius + CONTACT_SLACK


def _top_crown(fruits: list[Fruit]) -> float:
    """The topmost crown y. The floor if empty."""
    if not fruits:
        return float(NORMALIZED_HEIGHT)
    return min(f.y - f.radius for f in fruits)


def _bury_penalty(fruits: list[Fruit]) -> float:
    """How much a different type sits directly above a same type (0-)."""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type == under.type:
                continue
            if abs(over.x - under.x) > under.radius * 0.9:
                continue
            gap = (over.y - over.radius) - (under.y + under.radius)
            if -MERGE_SLACK <= gap <= under.radius * 0.6:
                siblings = sum(1 for f in fruits if f.type == under.type and f is not under)
                if siblings >= 1:
                    penalty += 1.0
                else:
                    penalty += 0.35
    return penalty


def _next_setup_at(
    before: list[Fruit],
    after: list[Fruit],
    x: float,
    next_type: int,
    merges: int,
) -> float:
    """High when the current drop column does not ruin being near / exposing a same type as next (0-1+)."""
    targets = [f for f in before if f.type == next_type]
    if not targets:
        return 0.0

    next_r = _radius(next_type)
    best = 0.0
    for fruit in targets:
        dist = abs(x - fruit.x)
        reach = fruit.radius + next_r + MERGE_SLACK
        if dist <= reach:
            proximity = 1.0 - dist / max(reach, 1.0)
        elif dist <= reach * 2.5:
            proximity = 0.35 * (1.0 - (dist - reach) / (reach * 1.5))
        else:
            proximity = 0.0

        buried = any(
            abs(f.x - fruit.x) <= fruit.radius * 0.9 and f.type != next_type and f.y < fruit.y
            for f in after
        )
        if buried:
            proximity *= 0.15
        best = max(best, proximity)

    if merges > 0:
        return best * 0.35
    return best


def _height_variance(fruits: list[Fruit]) -> float:
    """Spread of crowns per column bin. 0 if empty."""
    bins: dict[int, float] = {}
    for fruit in fruits:
        key = int(fruit.x // FLAT_BIN)
        top = fruit.y - fruit.radius
        bins[key] = min(bins.get(key, float(NORMALIZED_HEIGHT)), top)
    if len(bins) < 2:
        return 0.0
    return float(statistics.pstdev(list(bins.values())))


def _land_y(fruits: tuple[Fruit, ...] | list[Fruit], x: float, held_r: float) -> float:
    """Center y when dropped at column x. The floor, or on top of the crown of an overlapping fruit."""
    top = float(NORMALIZED_HEIGHT)
    for fruit in fruits:
        if abs(fruit.x - x) > fruit.radius + held_r:
            continue
        top = min(top, fruit.y - fruit.radius)
    return top - held_r


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
