"""Falling, rolling, collision pushes and merging. Shared by policy scoring and SimEnv.

policy calls this to score moves. The observed board is not assumed still, and
existing fruits are also moved by the momentum of hits and merges.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# Contact for a virtual merge. The observed board is assumed still, so do not loosen too much.
CONTACT_SLACK = 2.0
# Rolling after landing. If it sits on a side, shift sideways down to the valley.
SETTLE_STEP = 3.0
SETTLE_MAX_ITERS = 48
# Cap on how many times unsupported fruits fall after a merge.
BOARD_SETTLE_MAX_ITERS = 32
# Iterations pushing aside existing fruits through collisions and merges.
KNOCK_MAX_ITERS = 24
# Minimum push on contact. Below this it does not visibly move.
KNOCK_BASE = 18.0
# Push proportional to the moving side's radius (even a small fruit pushes by the other's radius).
KNOCK_RADIUS_FRAC = 1.15
# Upper limit of one push.
KNOCK_MAX = 80.0
# How much the merge result's spawn position shifts toward the momentum side.
MERGE_BIAS = 10.0
# |dx| / radius considered nearly at the top of the supporting circle (unstable).
APEX_DX_FRAC = 0.2


def simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int, list[int]]:
    """Board after dropping at column x, merge count and the list of source types merged."""
    placed = list(fruits)
    # Push aside fruits hit before falling (hit from the left, push right).
    placed, pre_knocked, pre_dir = _knock_drop_column(placed, fruit_type, x)
    placed, settled0 = _settle_board(placed)
    placed, dropped, coast_dir = _place(placed, fruit_type, x)
    # Prefer the direction slid on the floor. The push direction of a shoulder hit is the pre_dir side.
    if abs(coast_dir) > 0:
        knock_dir = coast_dir
    elif abs(pre_dir) > 0:
        knock_dir = pre_dir
    else:
        knock_dir = _impact_dir(placed, dropped, coast_dir)
    placed, knocked = _knock_contacts(placed, {dropped}, knock_dir)
    placed, settled = _settle_board(placed)
    active = {dropped} | pre_knocked | settled0 | knocked | settled
    return _resolve_merges(placed, active=active)


def preview_land(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
    held_r: float,
) -> tuple[float, float]:
    """Landing (x, y) after rolling, from drop column x."""
    x = _settle_x(fruits, x, held_r, allow_coast=True, drop_type=fruit_type)
    return x, land_y(fruits, x, held_r)


def land_y(fruits: tuple[Fruit, ...] | list[Fruit], x: float, held_r: float) -> float:
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


def _place(
    fruits: list[Fruit],
    fruit_type: int,
    x: float,
    *,
    allow_coast: bool = True,
) -> tuple[list[Fruit], int, float]:
    """Land a fruit at column x and add it.

    Returns (board, added index, direction of sideways movement until landing).
    """
    r = fruit_radius(fruit_type)
    x0 = max(r, min(NORMALIZED_WIDTH - r, x))
    x = _settle_x(fruits, x, r, allow_coast=allow_coast, drop_type=fruit_type)
    y = land_y(fruits, x, r)
    coast_dir = math.copysign(1.0, x - x0) if abs(x - x0) > 0.5 else 0.0
    fruits.append(Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=100.0))
    return fruits, len(fruits) - 1, coast_dir


def _place_on_floor(
    fruits: list[Fruit],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int]:
    """Place the merge result at a floor coordinate for now. Assumes settling after pushing aside."""
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    y = NORMALIZED_HEIGHT - r
    fruits.append(Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=100.0))
    return fruits, len(fruits) - 1


def _knock_drop_column(
    fruits: list[Fruit],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], set[int], float]:
    """Push aside existing fruits hit at the drop column in the direction of the hit."""
    fruits = list(fruits)
    if not fruits:
        return fruits, set(), 0.0
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    y = land_y(fruits, x, r)
    ghost = Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=100.0)
    fruits.append(ghost)
    ghost_i = len(fruits) - 1
    direction = 0.0
    best_dist = math.inf
    hit = False
    for j, b in enumerate(fruits[:-1]):
        # Same types are not pushed; merging handles them. Pushing first separates them and they do not merge.
        if b.type == fruit_type:
            continue
        dist = math.hypot(x - b.x, y - b.y)
        if dist > r + b.radius + CONTACT_SLACK:
            continue
        # Nearly directly on top is a lid or apex. Pushing exposes what is below and causes false merges, so only side contacts.
        if abs(b.x - x) <= max(b.radius * 0.25, r * 0.25):
            continue
        hit = True
        if dist < best_dist:
            best_dist = dist
            direction = math.copysign(1.0, b.x - x)
    if not hit:
        fruits.pop()
        return fruits, set(), 0.0
    if abs(direction) < 1e-9:
        direction = 1.0
    fruits, moved = _knock_contacts(fruits, {ghost_i}, direction)
    fruits.pop()
    moved = {i for i in moved if i < len(fruits)}
    return fruits, moved, direction


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
        y = land_y(fruits, x, held_r)
        if y >= floor - 1.0:
            if coast_dir == 0.0 or not allow_coast:
                return x
            return _coast_on_floor(fruits, x, held_r, coast_dir)

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
            if abs(apex_dx) > 1e-9:
                push = math.copysign(1.0, apex_dx)
            else:
                push = _apex_roll_dir(apex_support_x)

        coast_dir = math.copysign(1.0, push)
        nxt = max(lo, min(hi, x + coast_dir * SETTLE_STEP))
        nxt_y = land_y(fruits, nxt, held_r)
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
    """Whether it touches a same type at column (x, y). Shoulders included."""
    for fruit in fruits:
        if fruit.type != drop_type:
            continue
        if math.hypot(x - fruit.x, y - fruit.y) <= fruit.radius + held_r + CONTACT_SLACK:
            return True
    return False


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
        nxt_y = land_y(fruits, nxt, held_r)
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
    """Merge same-type contacts starting from the dropped fruit.

    The momentum of collisions and merges also pushes aside existing fruits. Fruits whose support vanished in a merge
    fall and roll, and only contacts continuing from moved fruits cascade.
    """
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
        knock_dir = _merge_knock_dir(i, j, a, b, active)
        for idx in sorted((i, j), reverse=True):
            fruits.pop(idx)

        merge_types.append(source_type)
        merges += 1
        fruits, moved = _settle_board(fruits)
        if new_type > MAX_FRUIT_TYPE:
            active = moved
            continue

        if abs(knock_dir) > 0:
            mid_x = mid_x + knock_dir * MERGE_BIAS
        # Place on the floor, then push the neighbors, then compute the landing height.
        fruits, new_i = _place_on_floor(fruits, new_type, mid_x)
        fruits, knocked = _knock_contacts(fruits, {new_i}, knock_dir)
        fruits, moved2 = _settle_board(fruits)
        active = {new_i} | moved | knocked | moved2

    return fruits, merges, merge_types


def _merge_knock_dir(
    i: int,
    j: int,
    a: Fruit,
    b: Fruit,
    active: set[int],
) -> float:
    """The push direction of a merge. +1 if active is right of the partner, -1 if left."""
    if i in active and j not in active:
        return math.copysign(1.0, a.x - b.x) if abs(a.x - b.x) > 1e-9 else 1.0
    if j in active and i not in active:
        return math.copysign(1.0, b.x - a.x) if abs(b.x - a.x) > 1e-9 else 1.0
    if abs(a.x - b.x) > 1e-9:
        return 1.0 if a.x >= b.x else -1.0
    return 1.0


def _impact_dir(fruits: list[Fruit], mover: int, coast_dir: float) -> float:
    """The direction a landing fruit pushes others. Inertia takes priority if present."""
    if abs(coast_dir) > 0:
        return coast_dir
    a = fruits[mover]
    best = 0.0
    best_dist = math.inf
    for j, b in enumerate(fruits):
        if j == mover:
            continue
        dist = math.hypot(a.x - b.x, a.y - b.y)
        limit = a.radius + b.radius
        if dist > limit + CONTACT_SLACK:
            continue
        if abs(b.x - a.x) <= 1e-9:
            continue
        if dist < best_dist:
            best_dist = dist
            best = math.copysign(1.0, b.x - a.x)
    return best


def _fruit_mass(fruit: Fruit) -> float:
    return max(1.0, fruit.radius * fruit.radius)


def _clamp_fruit_x(fruit: Fruit, x: float) -> float:
    return max(fruit.radius, min(NORMALIZED_WIDTH - fruit.radius, x))


def _knock_amount(mover: Fruit, overlap: float) -> float:
    """Push amount. Even light→heavy, move visibly on a radius basis."""
    by_radius = mover.radius * KNOCK_RADIUS_FRAC
    return min(KNOCK_MAX, max(overlap, KNOCK_BASE, by_radius))


def _knock_contacts(
    fruits: list[Fruit],
    movers: set[int],
    direction: float,
) -> tuple[list[Fruit], set[int]]:
    """Push existing fruits touched by movers aside.

    Hit from the left, push right; hit from the right, push left.
    The momentum push happens only once. After that, only untangle overlaps.
    """
    fruits = list(fruits)
    moved: set[int] = set()
    if not movers or not fruits:
        return fruits, moved

    impulses: list[tuple[int, int, float]] = []
    for i in movers:
        if i < 0 or i >= len(fruits):
            continue
        a = fruits[i]
        for j, b in enumerate(fruits):
            if j == i:
                continue
            dist = math.hypot(a.x - b.x, a.y - b.y)
            limit = a.radius + b.radius
            if dist > limit + CONTACT_SLACK:
                continue
            if abs(direction) > 0:
                if direction > 0 and b.x < a.x - 1.0:
                    continue
                if direction < 0 and b.x > a.x + 1.0:
                    continue
                push_dir = direction
            elif abs(b.x - a.x) > 1e-9:
                push_dir = math.copysign(1.0, b.x - a.x)
            else:
                push_dir = 1.0
            overlap = max(0.0, limit - dist)
            knock = _knock_amount(a, overlap)
            impulses.append((i, j, push_dir * knock))

    for i, j, raw in impulses:
        a = fruits[i]
        b = fruits[j]
        ma = _fruit_mass(a)
        mb = _fruit_mass(b)
        # The mass ratio only adjusts. The partner moves at least 55% of the push.
        share_b = max(0.55, ma / (ma + mb))
        push_dir = math.copysign(1.0, raw)
        knock = abs(raw)
        new_b = _clamp_fruit_x(b, b.x + push_dir * knock * share_b)
        new_a = _clamp_fruit_x(a, a.x - push_dir * knock * 0.05 * (mb / (ma + mb)))
        if abs(new_b - b.x) >= 0.25:
            fruits[j] = replace(b, x=new_b)
            moved.add(j)
        if abs(new_a - a.x) >= 0.25:
            fruits[i] = replace(a, x=new_a)
            moved.add(i)

    for _ in range(KNOCK_MAX_ITERS):
        fruits, sep = _separate_overlaps(fruits)
        if not sep:
            break
        moved |= sep

    return fruits, moved


def _separate_overlaps(fruits: list[Fruit]) -> tuple[list[Fruit], set[int]]:
    """Untangle circle overlaps left and right (1 pass)."""
    fruits = list(fruits)
    moved: set[int] = set()
    if len(fruits) < 2:
        return fruits, moved

    for i in range(len(fruits)):
        a = fruits[i]
        for j in range(i + 1, len(fruits)):
            b = fruits[j]
            dist = math.hypot(a.x - b.x, a.y - b.y)
            limit = a.radius + b.radius
            if dist >= limit - 0.25:
                continue
            if dist < 1e-9:
                sep_x = 1.0
            else:
                sep_x = (b.x - a.x) / dist
            if abs(sep_x) < 0.2:
                sep_x = 1.0 if b.x >= a.x else -1.0
            overlap = limit - dist
            ma = _fruit_mass(a)
            mb = _fruit_mass(b)
            total = ma + mb
            dx_a = -sep_x * overlap * (mb / total)
            dx_b = sep_x * overlap * (ma / total)
            na = _clamp_fruit_x(a, a.x + dx_a)
            nb = _clamp_fruit_x(b, b.x + dx_b)
            if abs(na - a.x) < 0.1 and abs(nb - b.x) < 0.1:
                continue
            fruits[i] = replace(a, x=na)
            fruits[j] = replace(b, x=nb)
            a = fruits[i]
            moved.add(i)
            moved.add(j)
    return fruits, moved


def _settle_board(fruits: list[Fruit]) -> tuple[list[Fruit], set[int]]:
    """Let unsupported fruits fall, and roll only fallen fruits off unstable tops.

    Indices are preserved (for tracking active merges).
    """
    fruits = list(fruits)
    moved_all: set[int] = set()
    if not fruits:
        return fruits, moved_all

    for _ in range(BOARD_SETTLE_MAX_ITERS):
        order = sorted(range(len(fruits)), key=lambda i: (-fruits[i].y, fruits[i].x))
        settled: list[int] = []
        fell: set[int] = set()
        for i in order:
            f = fruits[i]
            y = land_y([fruits[j] for j in settled], f.x, f.radius)
            if abs(y - f.y) > 0.5:
                fruits[i] = replace(f, y=y)
                fell.add(i)
            settled.append(i)

        rolled: set[int] = set()
        for i in fell:
            f = fruits[i]
            others = [fruits[j] for j in range(len(fruits)) if j != i]
            x2 = _settle_x(others, f.x, f.radius, allow_coast=True, drop_type=f.type)
            y2 = land_y(others, x2, f.radius)
            if abs(x2 - f.x) > 0.5 or abs(y2 - f.y) > 0.5:
                fruits[i] = replace(f, x=x2, y=y2)
                rolled.add(i)

        pass_moved = fell | rolled
        moved_all |= pass_moved
        if not pass_moved:
            break

    return fruits, moved_all


def _find_merge_pair(fruits: list[Fruit], active: set[int]) -> tuple[int, int] | None:
    """Same-type pairs in contact that may merge with the active side."""
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
