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
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    xs = {round(x / CANDIDATE_STEP) * CANDIDATE_STEP for x in _frange(lo, hi, CANDIDATE_STEP)}
    xs.add(_ideal_x(drop_type))

    for fruit in fruits:
        # Above / beside same types or slightly bigger fruits (orange→apple and so on).
        if fruit.type < drop_type or fruit.type > drop_type + 2:
            continue
        xs.add(fruit.x)
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)

    if extra_type is not None:
        xs.add(_ideal_x(extra_type))
        for fruit in fruits:
            if fruit.type != extra_type:
                continue
            xs.add(fruit.x)
            gap = held_r + fruit.radius
            xs.add(fruit.x - gap)
            xs.add(fruit.x + gap)

    return [x for x in xs if lo <= x <= hi]


def _score(obs: Observation, x: float, held_r: float) -> float:
    """Score the board after dropping held + the hypothetical best move of next."""
    assert obs.held_type is not None
    before = list(obs.fruits)
    after, merges = _simulate_drop(before, obs.held_type, x)
    land_y = _land_y(before, x, held_r)
    cleared_wedge = _clears_wedged(before, x, obs.held_type, held_r, merges)

    score = _board_score(after, merges, land_y=land_y)
    score += _wedged_priority(before, obs.held_type, cleared_wedge)
    score += _larger_neighbor_bonus(before, x, obs.held_type, held_r, land_y)
    # Moves merging a wedged same type do not force leaning toward the big fruit.
    if not cleared_wedge:
        score -= _ignored_larger_penalty(before, x, obs.held_type, held_r, land_y)

    # Without a merge, lean toward the 'lining-up side' of a one-tier-bigger fruit.
    if merges == 0:
        score -= abs(x - _anchor_x(obs.held_type, before, held_r)) * 0.45
        if not _column_fruits(before, x, held_r):
            score += 3.0

    if obs.next_type is not None:
        score += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)

    return score


def _best_next_score(fruits: list[Fruit], next_type: int) -> float:
    """Board score when next is dropped at its best column."""
    next_r = _radius(next_type)
    best = -math.inf
    for nx in _candidates(fruits, next_type, next_r):
        nx = clamp_drop_x(nx, next_type)
        after, merges = _simulate_drop(fruits, next_type, nx)
        land_y = _land_y(fruits, nx, next_r)
        cleared_wedge = _clears_wedged(fruits, nx, next_type, next_r, merges)
        value = _board_score(after, merges, land_y=land_y)
        value += _wedged_priority(fruits, next_type, cleared_wedge)
        value += _larger_neighbor_bonus(fruits, nx, next_type, next_r, land_y)
        if not cleared_wedge:
            value -= _ignored_larger_penalty(fruits, nx, next_type, next_r, land_y)
        if merges == 0:
            value -= abs(nx - _anchor_x(next_type, fruits, next_r)) * 0.45
        if value > best:
            best = value
    return 0.0 if best == -math.inf else best


def _is_wedged(fruit: Fruit, fruits: list[Fruit] | tuple[Fruit, ...]) -> bool:
    """Bigger fruits are close on both left and right, wedging it in."""
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


def _board_score(fruits: list[Fruit], merges: int, *, land_y: float) -> float:
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
    score -= _size_order_penalty(fruits)
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
) -> float:
    """Relation to a one-tier-bigger fruit. An open 'lining-up side' > on top > the opposite side.

    Size order is left = big, right = small. The orange puts the floor right of the apple first,
    and stacks on top only when the right is blocked.
    """
    supports = [f for f in fruits if 1 <= f.type - drop_type <= 2]
    if not supports:
        return 0.0

    best = 0.0
    for support in supports:
        gap = support.type - drop_type
        side_x = _ordered_side_x(support, drop_type, held_r)
        side_free = _side_slot_free(fruits, support, side_x, held_r)
        on_top = _is_on_top(support, x, held_r, land_y)
        beside = abs(x - side_x) <= max(held_r, MERGE_SLACK)

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
) -> float:
    """Penalty when a one-tier-bigger fruit exists but it is placed neither on the lining-up side nor on top."""
    supports = [f for f in fruits if f.type - drop_type == 1]
    if not supports:
        return 0.0

    for support in supports:
        side_x = _ordered_side_x(support, drop_type, held_r)
        if abs(x - side_x) <= max(held_r, MERGE_SLACK):
            return 0.0
        if _is_on_top(support, x, held_r, land_y):
            return 0.0
    return 110.0


def _ordered_side_x(support: Fruit, drop_type: int, held_r: float) -> float:
    """The column lining up next to it in size order. The small fruit goes right of the big one."""
    if drop_type < support.type:
        return support.x + support.radius + held_r
    return support.x - support.radius - held_r


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


def _ideal_x(fruit_type: int) -> float:
    """Bigger goes left. type 0 leans to the right edge."""
    return NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))


def _anchor_x(drop_type: int, fruits: list[Fruit] | tuple[Fruit, ...], held_r: float) -> float:
    """The column to place in. The lining-up side of a one-tier-bigger fruit if open, directly on top if blocked."""
    supports = [f for f in fruits if 1 <= f.type - drop_type <= 2]
    if not supports:
        return _ideal_x(drop_type)
    support = min(supports, key=lambda f: f.x)
    side_x = _ordered_side_x(support, drop_type, held_r)
    if _side_slot_free(fruits, support, side_x, held_r):
        return side_x
    return support.x


def _size_order_penalty(fruits: list[Fruit]) -> float:
    """Penalize pairs whose left-right size order is inverted. Looks at relative order rather than absolute ideal positions."""
    if not fruits:
        return 0.0
    penalty = 0.0
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            if left.type < right.type:
                penalty += (right.type - left.type) * 12.0
    penalty += sum(abs(f.x - _ideal_x(f.type)) for f in fruits) / len(fruits) * 0.12
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
    """How much merge candidates are buried by other types. Putting a small fruit on a big fruit is not penalized."""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type == under.type:
                continue
            if over.type < under.type:
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
