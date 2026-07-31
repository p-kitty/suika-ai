"""Decide the drop column. A thin bootstrap policy (the groundwork for RL).

It has no concrete procedures (push-ins, restoring pushes, cascade gap opening and the like).
It only looks at merging, dangerous height, burying, light size order and rolling accident prevention.
Growing into valleys of big fruits is limited to when the valley has a same type, or held and next are both
one smaller than the walls (other gap filling gets the usual penalties).
Moves are scored as eval = score (the real game's merge points) - penalties (penalties for accidents and bad moves).
"""

from __future__ import annotations

import math
import statistics

from .observe import Observation, clamp_drop_x
from .reward import merge_score
from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE, SPAWN_MAX_TYPE
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
# Discount for the next move.
NEXT_DISCOUNT = 0.55
# Rolling after landing. If it sits on a side, shift sideways down to the valley.
SETTLE_STEP = 3.0
SETTLE_MAX_ITERS = 48
# |dx| / radius considered nearly at the top of the supporting circle (unstable).
APEX_DX_FRAC = 0.2
# Penalties are scaled to balance with the real-game score (1-65). The only bonus is the real-game score.
# Penalty for aiming nearly at the center of a different type (even if it rolls to the floor). It breaks easily on the real machine.
FOREIGN_AIM_PENALTY = 10.0
# Penalty per excess fruit when there are 3 or more of the same type. Up to 2 waiting is OK.
EXCESS_SAME_WEIGHT = 20.0
# Penalty per type difference of an inverted size pair.
SIZE_ORDER_PAIR_WEIGHT = 1.5
# Penalty per mean distance from the ideal column (weak; not forcing a layout).
SIZE_ORDER_IDEAL_WEIGHT = 0.004
# A weak pull toward the ideal column.
IDEAL_PULL = 0.015
# Penalty per height stacked from the floor (the minimum to stop stacking).
LAND_HEIGHT_WEIGHT = 0.05
DANGER_CROWN_WEIGHT = 0.5
BURY_WEIGHT = 20.0
# Penalty for moves blocking a waiting same-type pair with a bigger fruit of another type (per type difference).
BURY_BLOCK_WEIGHT = 14.0
# The ratio when blocking from the shoulder rather than directly above.
BURY_SHOULDER_SCALE = 0.5
VARIANCE_WEIGHT = 0.08
VARIANCE_DANGER_SCALE = 0.15
WRONG_SIDE_BASE = 8.0
WRONG_SIDE_TYPE_WEIGHT = 2.0
COAST_DRIFT_WEIGHT = 0.08
COAST_FLOOR_BONUS = 8.0


def choose_x(obs: Observation) -> float:
    """Return the column to drop from the observation. Assumes ready with held_type present."""
    if obs.held_type is None:
        raise ValueError("no held_type")

    held_r = fruit_radius(obs.held_type)
    best_x = NORMALIZED_WIDTH / 2
    best_score = -math.inf

    for x in _candidates(obs.fruits, obs.held_type, held_r, extra_type=obs.next_type):
        x = clamp_drop_x(x, obs.held_type)
        score = _score(obs, x, held_r)
        if score > best_score:
            best_score = score
            best_x = x

    return best_x


def _candidates(
    fruits: tuple[Fruit, ...] | list[Fruit],
    drop_type: int,
    held_r: float,
    extra_type: int | None = None,
) -> list[float]:
    """Uniform spacing plus spots above / beside same-type and nearby fruits, and ideal_x."""
    sign = _order_sign(fruits)
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    xs = {round(x / CANDIDATE_STEP) * CANDIDATE_STEP for x in _frange(lo, hi, CANDIDATE_STEP)}
    xs.add(_ideal_x(drop_type, sign))

    for fruit in fruits:
        if fruit.type < drop_type or fruit.type > drop_type + 2:
            continue
        xs.add(fruit.x)
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)

    if extra_type is not None:
        xs.add(_ideal_x(extra_type, sign))
        for fruit in fruits:
            if fruit.type != extra_type:
                continue
            xs.add(fruit.x)
            gap = held_r + fruit.radius
            xs.add(fruit.x - gap)
            xs.add(fruit.x + gap)

    return [x for x in xs if lo <= x <= hi]


def drop_scores(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    *,
    next_type: int | None = None,
) -> tuple[float, float, float]:
    """(score, penalties, eval) of one move dropped at column x. For sim / training.

    Board penalties are the difference from before the drop. On the same board it is a constant difference, so choose_x's
    choice does not change, and summing it per move does not grow with the size of the board.
    """
    held_r = fruit_radius(drop_type)
    before = list(fruits)
    _, score, penalties = _evaluate_drop(
        before,
        drop_type,
        clamp_drop_x(x, drop_type),
        held_r,
        next_type=next_type,
    )
    penalties -= _board_penalties(before, sign=_order_sign(before))
    return score, penalties, score - penalties


def _score(obs: Observation, x: float, held_r: float) -> float:
    """Score the board after dropping held + the hypothetical best move of next."""
    assert obs.held_type is not None
    before = list(obs.fruits)
    after, score, penalties = _evaluate_drop(
        before, obs.held_type, x, held_r, next_type=obs.next_type
    )
    value = score - penalties
    if obs.next_type is not None:
        value += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)
    return value


def _best_next_score(fruits: list[Fruit], next_type: int) -> float:
    """eval when next is dropped at its best column."""
    next_r = fruit_radius(next_type)
    best = -math.inf
    for nx in _candidates(fruits, next_type, next_r):
        nx = clamp_drop_x(nx, next_type)
        # The next after that is unknown. Only same-type fruit in a valley counts for the growing exemption.
        _, score, penalties = _evaluate_drop(fruits, next_type, nx, next_r)
        if score - penalties > best:
            best = score - penalties
    return 0.0 if best == -math.inf else best


def _evaluate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    held_r: float,
    *,
    next_type: int | None = None,
) -> tuple[list[Fruit], float, float]:
    """Board, real-game score and penalties after one drop."""
    before = list(fruits)
    sign = _order_sign(before)
    land_x, land_y = _preview_land(before, drop_type, x, held_r)
    after, merges, merge_types = simulate_drop(before, drop_type, x)

    score = merge_score(merge_types)
    penalties = _board_penalties(after, sign=sign)
    # Only valley landings meeting the conditions are growing slots. Not crushed by height, wrong_side or ideal.
    growing = _valley_grow_ok(before, land_x, drop_type, next_type)
    if merges == 0:
        # A merged fruit does not remain, so the stacking penalty applies only to moves that stay on the board.
        if not growing:
            floor = NORMALIZED_HEIGHT - held_r
            penalties += max(0.0, floor - land_y) * LAND_HEIGHT_WEIGHT
            penalties += _wrong_side_roll_penalty(
                before, land_x, land_y, drop_type, held_r, sign
            )
            penalties += abs(x - _ideal_x(drop_type, sign)) * IDEAL_PULL
        penalties += _foreign_aim_penalty(before, x, drop_type)
        penalties += _bury_block_penalty(before, land_x, land_y, drop_type, held_r)
    penalties += _coast_away_penalty(before, x, land_x, land_y, held_r)
    return after, score, penalties


def _board_penalties(fruits: list[Fruit], *, sign: int = 1) -> float:
    """Board penalties after the drop (danger, burying, excess same type, size order, bumpiness)."""
    penalty = 0.0
    crown = _top_crown(fruits)
    if crown < DANGER_Y:
        penalty += (DANGER_Y - crown) * DANGER_CROWN_WEIGHT

    penalty += BURY_WEIGHT * _bury_penalty(fruits)
    penalty += _excess_same_penalty(fruits)
    penalty += _size_order_penalty(fruits, sign)
    variance = _height_variance(fruits)
    if crown < DANGER_Y:
        variance *= VARIANCE_DANGER_SCALE
    penalty += VARIANCE_WEIGHT * variance
    return penalty


def _excess_same_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """Penalize the excess when there are 3 or more of the same type. Up to 2 are allowed as waiting to merge."""
    counts: dict[int, int] = {}
    for fruit in fruits:
        counts[fruit.type] = counts.get(fruit.type, 0) + 1
    penalty = 0.0
    for count in counts.values():
        if count >= 3:
            penalty += (count - 2) * EXCESS_SAME_WEIGHT
    return penalty


def _foreign_aim_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_x: float,
    drop_type: int,
) -> float:
    """Penalty for aiming nearly at the center of a different type. Unstable on the real machine even if it rolls."""
    for fruit in fruits:
        if fruit.type == drop_type:
            continue
        if abs(drop_x - fruit.x) <= fruit.radius * 0.3:
            return FOREIGN_AIM_PENALTY
    return 0.0


def _wrong_side_roll_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
    sign: int,
) -> float:
    """Penalty for rolling onto the big-side floor of a big fruit."""
    floor = NORMALIZED_HEIGHT - held_r
    if land_y < floor - 4.0:
        return 0.0

    penalty = 0.0
    for other in fruits:
        if other.type <= drop_type:
            continue
        if (land_x - other.x) * sign >= 0:
            continue
        if abs(land_x - other.x) > other.radius + held_r + MERGE_SLACK * 2:
            continue
        penalty += WRONG_SIDE_BASE + WRONG_SIDE_TYPE_WEIGHT * (other.type - drop_type)
    return penalty


def _coast_away_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_x: float,
    land_x: float,
    land_y: float,
    held_r: float,
) -> float:
    """Penalize landings knocked far from the drop column by contact."""
    floor = NORMALIZED_HEIGHT - held_r
    drifted = abs(land_x - drop_x)
    if drifted < held_r * 2:
        return 0.0
    penalty = drifted * COAST_DRIFT_WEIGHT
    if land_y >= floor - 4.0 and drifted > NORMALIZED_WIDTH * 0.25:
        penalty += COAST_FLOOR_BONUS
    return penalty


def _valley_flanks(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
) -> tuple[Fruit, Fruit] | None:
    """Left and right when x is in a narrow valley between fruits bigger than drop_type."""
    left_big: Fruit | None = None
    right_big: Fruit | None = None
    for fruit in fruits:
        if fruit.type <= drop_type:
            continue
        if fruit.x < x:
            if left_big is None or fruit.x > left_big.x:
                left_big = fruit
        elif fruit.x > x:
            if right_big is None or fruit.x < right_big.x:
                right_big = fruit
    if left_big is None or right_big is None:
        return None
    held_r = fruit_radius(drop_type)
    sep = right_big.x - left_big.x
    touch = left_big.radius + right_big.radius
    if sep > touch + held_r * 2.8 + MERGE_SLACK:
        return None
    return left_big, right_big


def _valley_grow_ok(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    drop_type: int,
    next_type: int | None,
) -> bool:
    """Whether the growing exemption for a valley may apply.

    - there is a same type between the valley walls (cleanup, waiting to merge)
    - or held and next are both one smaller than the left and right walls
    """
    flanks = _valley_flanks(fruits, land_x, drop_type)
    if flanks is None:
        return False
    left, right = flanks
    for fruit in fruits:
        if fruit.type == drop_type and left.x < fruit.x < right.x:
            return True
    wall = min(left.type, right.type)
    return (
        next_type is not None
        and drop_type == next_type
        and drop_type == wall - 1
    )


def _is_nestled(
    fruit: Fruit,
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> bool:
    """Whether it sits in a valley between bigger fruits."""
    return _valley_flanks(fruits, fruit.x, fruit.type) is not None


def _ideal_x(fruit_type: int, sign: int = 1) -> float:
    """With sign=+1, bigger goes left. With sign=-1, bigger goes right."""
    base = NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))
    if sign < 0:
        return NORMALIZED_WIDTH - base
    return base


def _order_sign(fruits: list[Fruit] | tuple[Fruit, ...]) -> int:
    """The size direction of the board. +1 = big left, small right; -1 = small left, big right."""
    if not fruits:
        return 1
    if len(fruits) == 1:
        fruit = fruits[0]
        if fruit.type >= SPAWN_MAX_TYPE and fruit.x > NORMALIZED_WIDTH * 0.55:
            return -1
        return 1

    votes = 0.0
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if a.type == b.type:
                continue
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            weight = float(abs(a.type - b.type)) * (1.0 + 0.15 * max(a.type, b.type))
            if left.type > right.type:
                votes += weight
            else:
                votes -= weight

    if abs(votes) < 1.0:
        biggest = max(fruits, key=lambda f: (f.type, f.radius))
        return -1 if biggest.x > NORMALIZED_WIDTH * 0.5 else 1
    return 1 if votes > 0 else -1


def _size_order_penalty(fruits: list[Fruit], sign: int = 1) -> float:
    """Penalize pairs whose left-right size order is inverted. Looks at relative order rather than absolute ideal positions.

    Small fruits being grown in a valley are excluded from size order (layout penalties do not crush growing).
    """
    if not fruits:
        return 0.0
    penalty = 0.0
    open_fruits = [f for f in fruits if not _is_nestled(f, fruits)]
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            if _is_nestled(a, fruits) or _is_nestled(b, fruits):
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            if sign > 0 and left.type < right.type:
                penalty += (right.type - left.type) * SIZE_ORDER_PAIR_WEIGHT
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * SIZE_ORDER_PAIR_WEIGHT
    if open_fruits:
        penalty += (
            sum(abs(f.x - _ideal_x(f.type, sign)) for f in open_fruits)
            / len(open_fruits)
            * SIZE_ORDER_IDEAL_WEIGHT
        )
    return penalty


def simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int, list[int]]:
    """Board after dropping at column x, merge count and the list of source types merged. For sim / training."""
    placed = list(fruits)
    placed, dropped = _place(placed, fruit_type, x)
    return _resolve_merges(placed, active={dropped})


def _preview_land(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
    held_r: float,
) -> tuple[float, float]:
    """Landing (x, y) after rolling, from drop column x."""
    x = _settle_x(fruits, x, held_r, allow_coast=True, drop_type=fruit_type)
    return x, _land_y(fruits, x, held_r)


def _place(
    fruits: list[Fruit],
    fruit_type: int,
    x: float,
    *,
    allow_coast: bool = True,
) -> tuple[list[Fruit], int]:
    """Land a fruit at column x and add it. Also returns the added index."""
    r = fruit_radius(fruit_type)
    x = _settle_x(fruits, x, r, allow_coast=allow_coast, drop_type=fruit_type)
    y = _land_y(fruits, x, r)
    fruits.append(Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=100.0))
    return fruits, len(fruits) - 1


def _settle_x(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
    *,
    allow_coast: bool = True,
    drop_type: int | None = None,
) -> float:
    """If it sits on a circle's side, roll it down to the valley or floor, and on the floor slide by inertia to a wall / other fruit.

    Nearly at the top of a different type's supporting circle is unstable, so roll it to one side.
    Directly above the same type it lands as is, to merge.
    """
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    x = max(lo, min(hi, x))
    floor = NORMALIZED_HEIGHT - held_r
    coast_dir = 0.0

    for _ in range(SETTLE_MAX_ITERS):
        y = _land_y(fruits, x, held_r)
        if y >= floor - 1.0:
            if coast_dir == 0.0 or not allow_coast:
                return x
            return _coast_on_floor(fruits, x, held_r, coast_dir)

        # If the current column reaches a same type, do not roll it on a different type's slope or top and miss the merge.
        if drop_type is not None and _would_merge_at(fruits, x, y, held_r, drop_type):
            return x

        push = 0.0
        apex_dx = 0.0
        apex_support_x = x
        on_apex = False
        for fruit in fruits:
            dx = x - fruit.x
            gap = fruit.radius + held_r
            if abs(dx) >= gap - 1e-6:
                continue
            dy = math.sqrt(max(0.0, gap * gap - dx * dx))
            if abs((fruit.y - dy) - y) > 2.0:
                continue
            push += dx
            if drop_type is not None and fruit.type == drop_type:
                continue
            if abs(dx) <= max(fruit.radius * APEX_DX_FRAC, 1.0):
                on_apex = True
                apex_dx = dx
                apex_support_x = fruit.x

        if abs(push) < 0.75:
            if not on_apex:
                return x
            # Break a balance on the top. With a tiny dx use that direction, otherwise left or right deterministically.
            if abs(apex_dx) > 1e-9:
                push = math.copysign(1.0, apex_dx)
            else:
                push = _apex_roll_dir(apex_support_x)

        coast_dir = math.copysign(1.0, push)
        nxt = max(lo, min(hi, x + coast_dir * SETTLE_STEP))
        nxt_y = _land_y(fruits, nxt, held_r)
        if nxt_y < y - 0.5:
            return x
        if abs(nxt - x) < 1e-6:
            return x
        x = nxt
    return x


def _would_merge_at(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    y: float,
    held_r: float,
    drop_type: int,
) -> bool:
    """Whether a same type landed at column (x, y) touches an existing same type."""
    held = Fruit(type=drop_type, x=x, y=y, radius=held_r, confidence=100.0)
    return any(fruit.type == drop_type and _touching(held, fruit) for fruit in fruits)


def _apex_roll_dir(support_x: float) -> float:
    """The direction breaking a balance directly on top. Always the same for the same support_x."""
    return 1.0 if int(round(support_x / SETTLE_STEP)) % 2 == 0 else -1.0


def _coast_on_floor(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
    direction: float,
) -> float:
    """After falling off a slope onto the floor, slide in that direction until touching a wall or another fruit."""
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    floor = NORMALIZED_HEIGHT - held_r
    direction = math.copysign(1.0, direction)
    max_iters = int(NORMALIZED_WIDTH / SETTLE_STEP) + 5

    for _ in range(max_iters):
        nxt = max(lo, min(hi, x + direction * SETTLE_STEP))
        if abs(nxt - x) < 1e-6:
            return x
        nxt_y = _land_y(fruits, nxt, held_r)
        if nxt_y < floor - 1.0:
            return x
        for fruit in fruits:
            limit = fruit.radius + held_r
            if abs(nxt - fruit.x) < limit - 0.5:
                if direction > 0:
                    return max(lo, min(hi, fruit.x - limit))
                return max(lo, min(hi, fruit.x + limit))
        x = nxt
    return x


def _resolve_merges(
    fruits: list[Fruit], active: set[int]
) -> tuple[list[Fruit], int, list[int]]:
    """Merge only same-type contacts starting from the dropped fruit. The observed board is assumed still."""
    fruits = list(fruits)
    merges = 0
    merge_types: list[int] = []
    for _ in range(64):
        pair = _find_merge_pair(fruits, active)
        if pair is None:
            break
        i, j = pair
        a, b = fruits[i], fruits[j]
        source_type = a.type
        new_type = source_type + 1
        mid_x = (a.x + b.x) / 2
        for idx in sorted((i, j), reverse=True):
            fruits.pop(idx)

        merge_types.append(source_type)
        merges += 1
        if new_type > MAX_FRUIT_TYPE:
            active = set()
            continue

        fruits, new_i = _place(fruits, new_type, mid_x, allow_coast=False)
        active = {new_i}

    return fruits, merges, merge_types


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
    """How much merge candidates are buried by other types."""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type == under.type:
                continue
            if over.type < under.type:
                continue
            if over.y >= under.y:
                continue
            if abs(over.x - under.x) > under.radius * 0.9:
                continue
            gap = (under.y - under.radius) - (over.y + over.radius)
            if -MERGE_SLACK <= gap <= under.radius * 0.6:
                siblings = sum(1 for f in fruits if f.type == under.type and f is not under)
                if siblings >= 1:
                    penalty += 1.0
                else:
                    penalty += 0.35
    return penalty


def _bury_block_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
) -> float:
    """Penalty for blocking a fruit waiting for a same-type pair with a bigger fruit of another type, from directly above or the shoulder."""
    penalty = 0.0
    for under in fruits:
        if under.type >= drop_type:
            continue
        if not any(f.type == under.type and f is not under for f in fruits):
            continue
        dx = abs(land_x - under.x)
        if dx > under.radius + held_r * 0.5:
            continue
        # Whether it sits on the head. Merely side by side does not block.
        over_top = (land_y + held_r) - (under.y - under.radius)
        if over_top > under.radius:
            continue
        scale = 1.0 if dx <= under.radius * 0.5 else BURY_SHOULDER_SCALE
        penalty += scale * BURY_BLOCK_WEIGHT * (drop_type - under.type)
    return penalty


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
    """Center y when dropped at column x. The floor, or where the circles touch."""
    best = float(NORMALIZED_HEIGHT) - held_r
    for fruit in fruits:
        dx = abs(fruit.x - x)
        gap = fruit.radius + held_r
        if dx >= gap:
            continue
        dy = math.sqrt(gap * gap - dx * dx)
        best = min(best, fruit.y - dy)
    return best


def _frange(start: float, stop: float, step: float):
    x = start
    while x <= stop + 1e-6:
        yield x
        x += step
