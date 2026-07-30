"""Decide the drop column. Size order + held/next heuristics."""

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
# Discount for the next move.
NEXT_DISCOUNT = 0.55
# Penalty per missing px when the columns of intermediate stages between big and small get crushed.
CHAIN_SPACING_WEIGHT = 2.0
# Rolling after landing. If it sits on a side, shift sideways down to the valley.
SETTLE_STEP = 3.0
SETTLE_MAX_ITERS = 48
# When lining up next to it, exact contact rides onto the shoulder and gets knocked, so leave a small gap.
SIDE_CLEARANCE = 4.0


def choose_x(obs: Observation) -> float:
    """Return the column to drop from the observation. Assumes ready with held_type present."""
    if obs.held_type is None:
        raise ValueError("no held_type")

    held_r = _radius(obs.held_type)
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
    """Uniform spacing plus spots above / beside same-type and one-tier-bigger fruits, and ideal_x."""
    sign = _order_sign(fruits)
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    xs = {round(x / CANDIDATE_STEP) * CANDIDATE_STEP for x in _frange(lo, hi, CANDIDATE_STEP)}
    xs.add(_ideal_x(drop_type, sign))

    for fruit in fruits:
        # Above / beside same types or slightly bigger fruits (orange→apple and so on).
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

    # Positions that leave columns for intermediate stages between small fruits on the small side / big fruits on the big side.
    for fruit in fruits:
        if fruit.type < drop_type:
            xs.add(fruit.x - sign * _chain_center_gap(drop_type, fruit.type))
            xs.add(fruit.x - sign * (_chain_center_gap(drop_type, fruit.type) + SIDE_CLEARANCE))
        elif fruit.type > drop_type:
            xs.add(fruit.x + sign * _chain_center_gap(fruit.type, drop_type))
            xs.add(fruit.x + sign * (_chain_center_gap(fruit.type, drop_type) + SIDE_CLEARANCE))

    beside = _smaller_neighbor_x(fruits, drop_type, held_r, sign)
    if beside is not None:
        xs.add(beside)

    return [x for x in xs if lo <= x <= hi]


def _score(obs: Observation, x: float, held_r: float) -> float:
    """Score the board after dropping held + the hypothetical best move of next."""
    assert obs.held_type is not None
    before = list(obs.fruits)
    sign = _order_sign(before)
    land_x, land_y = _preview_land(before, obs.held_type, x, held_r)
    after, merges = _simulate_drop(before, obs.held_type, x)
    cleared_wedge = _clears_wedged(before, land_x, obs.held_type, held_r, merges)
    grow_target = _growth_target_type(obs.held_type, obs.next_type)

    score = _board_score(after, merges, land_y=land_y, sign=sign)
    score += _wedged_priority(before, obs.held_type, cleared_wedge)
    score += _larger_neighbor_bonus(
        before,
        land_x,
        obs.held_type,
        held_r,
        land_y,
        grow_target=grow_target,
        sign=sign,
    )
    # Moves merging a wedged same type do not force leaning toward the big fruit.
    if not cleared_wedge:
        score -= _ignored_larger_penalty(
            before, land_x, obs.held_type, held_r, land_y, sign=sign
        )

    # Moves where hitting the shoulder rolls a small fruit onto the big side are dropped hard.
    # (e.g. upper left of a grape → rolls left → size order breaks)
    # Moves knocked far from the drop column are penalized too, even if they merge.
    score -= _wrong_side_roll_penalty(
        before, land_x, land_y, obs.held_type, held_r, sign
    ) if merges == 0 else 0.0
    score -= _coast_away_penalty(before, x, land_x, land_y, held_r)

    # Without a merge, lean toward the 'lining-up side' of a one-tier-bigger fruit.
    # Only when lining up on the floor, penalize crushing the columns of intermediate stages (x of stacking is out of scope).
    # Moves putting it on the target when growing get no pull toward the lining-up side.
    if merges == 0:
        on_grow = grow_target is not None and any(
            f.type == grow_target and _is_on_top(f, land_x, held_r, land_y) for f in before
        )
        if not on_grow:
            score -= abs(x - _anchor_x(obs.held_type, before, held_r, sign)) * 0.45
        floor = NORMALIZED_HEIGHT - held_r
        if land_y >= floor - 4.0 and not on_grow:
            score -= _chain_spacing_penalty(before, land_x, obs.held_type, sign)
        if not _column_fruits(before, x, held_r):
            score += 3.0

    if obs.next_type is not None:
        score += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)

    return score


def _growth_target_type(held_type: int, next_type: int | None) -> int | None:
    """If held and next are the same type, two of them can grow a one-tier-bigger fruit. That growing target."""
    if next_type is None or next_type != held_type:
        return None
    target = held_type + 1
    if target > MAX_FRUIT_TYPE:
        return None
    return target


def _best_next_score(fruits: list[Fruit], next_type: int) -> float:
    """Board score when next is dropped at its best column."""
    next_r = _radius(next_type)
    sign = _order_sign(fruits)
    best = -math.inf
    for nx in _candidates(fruits, next_type, next_r):
        nx = clamp_drop_x(nx, next_type)
        land_x, land_y = _preview_land(fruits, next_type, nx, next_r)
        after, merges = _simulate_drop(fruits, next_type, nx)
        cleared_wedge = _clears_wedged(fruits, land_x, next_type, next_r, merges)
        value = _board_score(after, merges, land_y=land_y, sign=sign)
        value += _wedged_priority(fruits, next_type, cleared_wedge)
        value += _larger_neighbor_bonus(
            fruits, land_x, next_type, next_r, land_y, sign=sign
        )
        if not cleared_wedge:
            value -= _ignored_larger_penalty(
                fruits, land_x, next_type, next_r, land_y, sign=sign
            )
        if merges == 0:
            value -= _wrong_side_roll_penalty(
                fruits, land_x, land_y, next_type, next_r, sign
            )
            on_grow = any(
                f.type == next_type + 1 and _is_on_top(f, land_x, next_r, land_y)
                for f in fruits
            ) and next_type + 1 <= MAX_FRUIT_TYPE
            # next alone has no held/next same-type growing flag, so only on top of a one-tier-bigger fruit is exempt.
            if not on_grow:
                value -= abs(nx - _anchor_x(next_type, fruits, next_r, sign)) * 0.45
                floor = NORMALIZED_HEIGHT - next_r
                if land_y >= floor - 4.0:
                    value -= _chain_spacing_penalty(fruits, land_x, next_type, sign)
            if not _column_fruits(fruits, nx, next_r):
                value += 3.0
        value -= _coast_away_penalty(fruits, nx, land_x, land_y, next_r)
        if value > best:
            best = value
    return 0.0 if best == -math.inf else best


def _chain_center_gap(left_type: int, right_type: int) -> float:
    """The center distance when lining up every intermediate stage between the big side and the small side."""
    gap = _radius(left_type) + _radius(right_type)
    for mid in range(right_type + 1, left_type):
        gap += 2.0 * _radius(mid)
    return gap


def _chain_spacing_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    sign: int = 1,
) -> float:
    """In a size-ordered row, penalize placements that crush the gap for intermediate stages.

    sign=+1 means left = big, right = small. sign=-1 is the reverse.
    """
    penalty = 0.0
    for other in fruits:
        if other.type < drop_type:
            # Small fruits belong on the small side.
            on_small_side = (other.x - x) * sign > 0
            if not on_small_side:
                continue
            need = _chain_center_gap(drop_type, other.type)
            lo_x, hi_x = (x, other.x) if x < other.x else (other.x, x)
            for mid in range(other.type + 1, drop_type):
                if any(lo_x < f.x < hi_x and f.type == mid for f in fruits):
                    need -= 2.0 * _radius(mid)
            have = abs(other.x - x)
            if have < need:
                penalty += (need - have) * CHAIN_SPACING_WEIGHT
        elif other.type > drop_type:
            on_large_side = (other.x - x) * sign < 0
            if not on_large_side:
                continue
            need = _chain_center_gap(other.type, drop_type)
            lo_x, hi_x = (x, other.x) if x < other.x else (other.x, x)
            for mid in range(drop_type + 1, other.type):
                if any(lo_x < f.x < hi_x and f.type == mid for f in fruits):
                    need -= 2.0 * _radius(mid)
            have = abs(other.x - x)
            if have < need:
                penalty += (need - have) * CHAIN_SPACING_WEIGHT
    return penalty


def _is_wedged(fruit: Fruit, fruits: list[Fruit] | tuple[Fruit, ...]) -> bool:
    """Bigger fruits are close on both left and right, wedging it in.

    Merely sitting on top of a big fruit is not wedged.
    """
    for other in fruits:
        if other is fruit or other.type <= fruit.type:
            continue
        if _is_on_top(other, fruit.x, fruit.radius, fruit.y):
            return False

    left_big: Fruit | None = None
    right_big: Fruit | None = None
    for other in fruits:
        if other is fruit or other.type <= fruit.type:
            continue
        if abs(other.y - fruit.y) > (other.radius + fruit.radius) * 1.5:
            continue
        reach = other.radius + fruit.radius + MERGE_SLACK
        dx = other.x - fruit.x
        if -reach * 1.25 <= dx < 0:
            if left_big is None or other.x > left_big.x:
                left_big = other
        elif 0 < dx <= reach * 1.25:
            if right_big is None or other.x < right_big.x:
                right_big = other
    return left_big is not None and right_big is not None


def _clears_wedged(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    held_r: float,
    merges: int,
) -> bool:
    """Whether the drop merges a wedged same type."""
    if merges < 1:
        return False
    for fruit in fruits:
        if fruit.type != drop_type or not _is_wedged(fruit, fruits):
            continue
        if abs(x - fruit.x) <= fruit.radius + held_r + MERGE_SLACK:
            return True
    return False


def _wedged_priority(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    cleared_wedge: bool,
) -> float:
    """A same type wedged by big fruits is grown before ordering."""
    has_wedge = any(f.type == drop_type and _is_wedged(f, fruits) for f in fruits)
    if not has_wedge:
        return 0.0
    if cleared_wedge:
        return 220.0
    return -220.0


def _board_score(
    fruits: list[Fruit],
    merges: int,
    *,
    land_y: float,
    sign: int = 1,
) -> float:
    """Board evaluation for one move (merges, height, burying, size order)."""
    score = 0.0
    score += 140.0 * merges
    if merges >= 2:
        score += 80.0 * (merges - 1)

    score += land_y * 0.22

    crown = _top_crown(fruits)
    score += crown * 0.8
    if crown < DANGER_Y:
        score -= (DANGER_Y - crown) * 4.0

    score -= 90.0 * _bury_penalty(fruits)
    score -= _size_order_penalty(fruits, sign)
    # When there is a dangerous pile, go low rather than flattening. Do not go 'evening out' heights by placing beside it.
    variance = _height_variance(fruits)
    if crown < DANGER_Y:
        variance *= 0.15
    score -= 1.2 * variance
    return score


def _larger_neighbor_bonus(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    held_r: float,
    land_y: float,
    *,
    grow_target: int | None = None,
    sign: int = 1,
) -> float:
    """Relation to a one-tier-bigger fruit. An open 'lining-up side' > on top > the opposite side.

    The neighbor is chosen according to the size-order direction (sign). With sign=+1, small goes right of big.

    But in positions where held and next are the same type and grow the one-tier-bigger fruit,
    'on top' of it takes priority over the lining-up side (put the second on it, merge → grow).
    """
    supports = [f for f in fruits if 1 <= f.type - drop_type <= 2]
    if not supports:
        return 0.0

    best = 0.0
    for support in supports:
        gap = support.type - drop_type
        side_x = _ordered_side_x(support, drop_type, held_r, sign)
        side_free = _side_slot_free(fruits, support, side_x, held_r)
        on_top = _is_on_top(support, x, held_r, land_y)
        beside = abs(x - side_x) <= max(held_r, MERGE_SLACK)
        growing = grow_target is not None and support.type == grow_target

        if growing and on_top:
            # On top of the target grown with a same-type next. Stronger than the lining-up side + low landing.
            best = max(best, 330.0 if gap == 1 else 150.0)
            continue

        if beside and side_free:
            best = max(best, 200.0 if gap == 1 else 90.0)
            continue

        if on_top:
            if side_free:
                # On top is weak when the neighbor is open.
                best = max(best, 35.0 if gap == 1 else 15.0)
            else:
                best = max(best, 150.0 if gap == 1 else 70.0)
            continue

        if abs(x - support.x) <= support.radius + held_r + MERGE_SLACK:
            best = max(best, 25.0 if gap == 1 else 10.0)

    return best


def _ignored_larger_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    held_r: float,
    land_y: float,
    *,
    sign: int = 1,
) -> float:
    """Penalty when a one-tier-bigger fruit exists but it is placed neither on the lining-up side nor on top."""
    supports = [f for f in fruits if f.type - drop_type == 1]
    if not supports:
        return 0.0

    for support in supports:
        side_x = _ordered_side_x(support, drop_type, held_r, sign)
        if abs(x - side_x) <= max(held_r, MERGE_SLACK):
            return 0.0
        if _is_on_top(support, x, held_r, land_y):
            return 0.0
    return 110.0


def _wrong_side_roll_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
    sign: int,
) -> float:
    """Penalty for rolling onto the big-side floor of a big fruit.

    Prevents breakage such as placing on a grape's left shoulder → rolling left → strawberry left of the grape.
    """
    floor = NORMALIZED_HEIGHT - held_r
    if land_y < floor - 4.0:
        return 0.0

    penalty = 0.0
    for other in fruits:
        if other.type <= drop_type:
            continue
        # With sign=+1, small is right of big. If land is left of big (the big side), it is broken.
        if (land_x - other.x) * sign >= 0:
            continue
        # Only when it rolls off the shoulder and lands right beside. Unrelated big fruits far away are not looked at.
        if abs(land_x - other.x) > other.radius + held_r + MERGE_SLACK * 2:
            continue
        penalty += 180.0 + 40.0 * (other.type - drop_type)
    return penalty


def _coast_away_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_x: float,
    land_x: float,
    land_y: float,
    held_r: float,
) -> float:
    """Penalize landings knocked far from the drop column by contact.

    Moves like meaning to go just left of a strawberry but sliding to the left edge.
    """
    floor = NORMALIZED_HEIGHT - held_r
    drifted = abs(land_x - drop_x)
    if drifted < held_r * 2:
        return 0.0
    # The farther it slid on the floor, the bigger the penalty.
    penalty = drifted * 1.4
    if land_y >= floor - 4.0 and drifted > NORMALIZED_WIDTH * 0.25:
        penalty += 120.0
    return penalty


def _ordered_side_x(
    support: Fruit,
    drop_type: int,
    held_r: float,
    sign: int = 1,
) -> float:
    """The column lining up next to it in size order. With sign=+1, the small fruit goes right of the big one.

    Exactly the sum of radii tends to ride onto the shoulder while falling and get knocked, so leave a small gap.
    """
    gap = support.radius + held_r + SIDE_CLEARANCE
    if drop_type < support.type:
        return support.x + sign * gap
    return support.x - sign * gap


def _smaller_neighbor_x(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    held_r: float,
    sign: int,
) -> float | None:
    """The column lining up right on the big side of a small fruit on the small side. None if there is none."""
    smallers = [f for f in fruits if f.type < drop_type]
    if not smallers:
        return None
    # The small fruit closest to the big side (with sign=+1, the leftmost small fruit).
    neighbor = min(smallers, key=lambda f: f.x * sign)
    gap = _chain_center_gap(drop_type, neighbor.type) + SIDE_CLEARANCE
    return neighbor.x - sign * gap


def _side_slot_free(
    fruits: list[Fruit] | tuple[Fruit, ...],
    support: Fruit,
    side_x: float,
    held_r: float,
) -> bool:
    """Whether the floor on the lining-up side is open (no obstruction other than the support)."""
    if side_x < held_r or side_x > NORMALIZED_WIDTH - held_r:
        return False
    land = _land_y_excluding(fruits, side_x, held_r, exclude=support)
    floor = NORMALIZED_HEIGHT - held_r
    return land >= floor - 4.0


def _land_y_excluding(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
    *,
    exclude: Fruit,
) -> float:
    return _land_y((f for f in fruits if f is not exclude), x, held_r)


def _is_on_top(support: Fruit, x: float, held_r: float, land_y: float) -> bool:
    """Whether it lands nearly directly above support."""
    if abs(x - support.x) > support.radius * 0.85:
        return False
    top = support.y - support.radius
    return abs((land_y + held_r) - top) <= MERGE_SLACK


def _ideal_x(fruit_type: int, sign: int = 1) -> float:
    """With sign=+1, bigger goes left. With sign=-1, bigger goes right."""
    base = NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))
    if sign < 0:
        return NORMALIZED_WIDTH - base
    return base


def _order_sign(fruits: list[Fruit] | tuple[Fruit, ...]) -> int:
    """The size direction of the board. +1 = big left, small right; -1 = small left, big right.

    A check so the left is not re-grown big when big fruits have already gathered on the right.
    """
    if not fruits:
        return 1
    if len(fruits) == 1:
        fruit = fruits[0]
        if fruit.type >= 4 and fruit.x > NORMALIZED_WIDTH * 0.55:
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


def _anchor_x(
    drop_type: int,
    fruits: list[Fruit] | tuple[Fruit, ...],
    held_r: float,
    sign: int = 1,
) -> float:
    """The column to place in. The lining-up side of a one-tier-bigger fruit if open, directly on top if blocked.

    With no big support, prefer right next to a small fruit on the small side
    (avoiding knock-aways / being too far from leaving space up to ideal).
    """
    supports = [f for f in fruits if 1 <= f.type - drop_type <= 2]
    if supports:
        # Prefer a support on the big side (left for sign=+1, right for sign=-1).
        support = min(supports, key=lambda f: f.x * sign)
        side_x = _ordered_side_x(support, drop_type, held_r, sign)
        if _side_slot_free(fruits, support, side_x, held_r):
            return side_x
        return support.x

    beside = _smaller_neighbor_x(fruits, drop_type, held_r, sign)
    if beside is not None:
        return max(held_r, min(beside, NORMALIZED_WIDTH - held_r))

    return max(held_r, min(_ideal_x(drop_type, sign), NORMALIZED_WIDTH - held_r))


def _size_order_penalty(fruits: list[Fruit], sign: int = 1) -> float:
    """Penalize pairs whose left-right size order is inverted. Looks at relative order rather than absolute ideal positions."""
    if not fruits:
        return 0.0
    penalty = 0.0
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            # sign=+1: the left should be bigger. sign=-1: the left should be smaller.
            if sign > 0 and left.type < right.type:
                penalty += (right.type - left.type) * 12.0
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * 12.0
    penalty += sum(abs(f.x - _ideal_x(f.type, sign)) for f in fruits) / len(fruits) * 0.12
    return penalty


def _after_drop(obs: Observation, x: float) -> tuple[list[Fruit], int]:
    """For tests. Board and merge count after dropping held at column x."""
    assert obs.held_type is not None
    return _simulate_drop(obs.fruits, obs.held_type, x)


def _simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int]:
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
    x = _settle_x(fruits, x, held_r, allow_coast=True)
    return x, _land_y(fruits, x, held_r)


def _place(
    fruits: list[Fruit],
    fruit_type: int,
    x: float,
    *,
    allow_coast: bool = True,
) -> tuple[list[Fruit], int]:
    """Land a fruit at column x and add it. Also returns the added index.

    Placed after including rolling after a side hit. Fruits produced by a merge
    get no inertial sliding (they settle near the midpoint).
    """
    r = _radius(fruit_type)
    x = _settle_x(fruits, x, r, allow_coast=allow_coast)
    y = _land_y(fruits, x, r)
    fruits.append(Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=100.0))
    return fruits, len(fruits) - 1


def _settle_x(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
    *,
    allow_coast: bool = True,
) -> float:
    """If it sits on a circle's side, roll it down to the valley or floor, and on the floor slide by inertia to a wall / other fruit."""
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

        push = 0.0
        for fruit in fruits:
            dx = x - fruit.x
            gap = fruit.radius + held_r
            if abs(dx) >= gap - 1e-6:
                continue
            dy = math.sqrt(max(0.0, gap * gap - dx * dx))
            if abs((fruit.y - dy) - y) > 2.0:
                continue
            # If it sits right of the pivot it keeps rolling off to the right.
            push += dx

        if abs(push) < 0.75:
            return x

        coast_dir = math.copysign(1.0, push)
        nxt = max(lo, min(hi, x + coast_dir * SETTLE_STEP))
        nxt_y = _land_y(fruits, nxt, held_r)
        # y points down. Stop when it gets smaller, since that is climbing.
        if nxt_y < y - 0.5:
            return x
        if abs(nxt - x) < 1e-6:
            return x
        x = nxt
    return x


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
    # A length that crosses an open floor to reach the wall.
    max_iters = int(NORMALIZED_WIDTH / SETTLE_STEP) + 5

    for _ in range(max_iters):
        nxt = max(lo, min(hi, x + direction * SETTLE_STEP))
        if abs(nxt - x) < 1e-6:
            return x
        nxt_y = _land_y(fruits, nxt, held_r)
        # Stop just before riding up onto another fruit's slope.
        if nxt_y < floor - 1.0:
            return x
        # If it sinks into another fruit on the floor, move to the contact position.
        for fruit in fruits:
            limit = fruit.radius + held_r
            if abs(nxt - fruit.x) < limit - 0.5:
                if direction > 0:
                    return max(lo, min(hi, fruit.x - limit))
                return max(lo, min(hi, fruit.x + limit))
        x = nxt
    return x


def _resolve_merges(fruits: list[Fruit], active: set[int]) -> tuple[list[Fruit], int]:
    """Merge only same-type contacts starting from the dropped fruit. The observed board is assumed still.

    The merged fruit appears at the midpoint and then rolls to its landing via `_place`.
    """
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

        fruits, new_i = _place(fruits, new_type, mid_x, allow_coast=False)
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
    """How much merge candidates are buried by other types. Putting a small fruit on a big fruit is not penalized."""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type == under.type:
                continue
            if over.type < under.type:
                continue
            # y points down. The burying side over is above under (smaller y).
            if over.y >= under.y:
                continue
            if abs(over.x - under.x) > under.radius * 0.9:
                continue
            # The gap between the top of under and the bottom of over.
            gap = (under.y - under.radius) - (over.y + over.radius)
            if -MERGE_SLACK <= gap <= under.radius * 0.6:
                siblings = sum(1 for f in fruits if f.type == under.type and f is not under)
                if siblings >= 1:
                    penalty += 1.0
                else:
                    penalty += 0.35
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
    """Center y when dropped at column x. The floor, or where the circles touch.

    With a sideways offset it slides along a big fruit's side, so it reaches small fruits fallen into gaps.
    """
    best = float(NORMALIZED_HEIGHT) - held_r
    for fruit in fruits:
        dx = abs(fruit.x - x)
        gap = fruit.radius + held_r
        if dx >= gap:
            continue
        dy = math.sqrt(gap * gap - dx * dx)
        best = min(best, fruit.y - dy)
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
