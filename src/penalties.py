"""The penalty side of scoring a move.

Called from the search in `policy.py`. Of eval = score (the real game's merge points) - penalties,
the side that counts the shapes this policy treats as 'accidents and bad moves' lives here.

Dependency is one-way (policy -> penalties). Do not call the search side from here.
The size direction of the board (`sign`) is decided by the caller and passed as an argument.

The comment on each constant records the measurement that settled that value. Before changing a number,
put it through the A/B in scripts/compare_policy.py. `_apply_variant`
rewrites module attributes, so references in here must always be read
through module globals (binding with `from .penalties import X` makes
the rewrite ineffective).
"""

from __future__ import annotations

import math
import statistics

from .observe import clamp_drop_x
from .sim.sim_physics import landed_xy, simulate_drop_held
from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE, SPAWN_MAX_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# --- Tuning shared across several places ---
# Tolerance for a contact that could merge (difference between center distance and sum of radii).
MERGE_SLACK = 18.0
# Sideways offset of a landing counted as directly above a different type (ratio to the lower fruit's radius).
FOREIGN_AIM_CENTER_FRAC = 0.20
# Penalty for landing in the center band of a different type directly below.
FOREIGN_AIM_PENALTY = 100.0
# Valley-growing bonus (applied by subtracting from penalties). Only for landings where `valley_grow_ok` holds.
# Not stronger than a real merge. At 8.0 it rejected a grape merge (6 points) for a non-merging valley.
# At 2.0 it tips toward growing, and at 3.0 it still keeps taking merges (measured).
VALLEY_GROW_BONUS = 3.0

# --- Wall-anchored check (shared by the corner pocket penalty and the ladder base) ---
EDGE_ANCHOR_MIN = 24.0
EDGE_ANCHOR_FRAC = 0.35

# Number of tiers below the biggest fruit counted as 'the big-fruit cluster'.
BIG_CLUSTER_SPAN = 2

# --- How full the floor is ---
# Height considered on the floor. A floor placement if the bottom is within this multiple of the radius.
FLOOR_BAND = 1.35
# Upper limit of a gap considered filled = the orange's diameter.
# It need not be connected from wall to wall; if an orange does not fit the gap,
# moves dropping there are not a problem, so it counts as filled.
FLOOR_PACKED_GAP = fruit_radius(SPAWN_MAX_TYPE) * 2.0

# --- Big draws after the floor fills ---
# When the floor fills there is no place left on the small side. Still ideal_x keeps pulling small fruits
# to the small side (orange's ideal is 236 = right side), so larger draws get stacked
# on the small side, crushing the small fruits below and collapsing. Once the floor fills, put them on the big side's shoulder
# instead of side by side. The ladder shape comes out as a result of this placement split.
# Measured (10 seeds × 120 moves): of 358 cases, 211 were placed on the small side, and in 210 of them
# eval really chose the small side (median +4.1). A problem of evaluation, not candidates.
PACKED_BIG_DRAW_MIN_TYPE = SPAWN_MAX_TYPE - 1
# It flips the narrow median margin of +4.1 while keeping moves that can actually merge on the small side (max +159.9).
# It is not applied to merging moves (only when merges == 0), so it does not compete with merging.
PACKED_SMALL_SIDE_WEIGHT = 8.0


# --- Helpers that only read the board -------------------------------------------------


def ideal_x(fruit_type: int, sign: int = 1) -> float:
    """With sign=+1, bigger goes left. With sign=-1, bigger goes right."""
    base = NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))
    if sign < 0:
        return NORMALIZED_WIDTH - base
    return base


def wall_gap(fruit: Fruit, sign: int) -> float:
    """Gap to the wall on the big side (sign)."""
    if sign > 0:
        return fruit.x - fruit.radius
    return NORMALIZED_WIDTH - fruit.radius - fruit.x


def is_wall_anchored(fruit: Fruit, sign: int) -> bool:
    """Whether it is on the big-side wall."""
    limit = max(EDGE_ANCHOR_MIN, fruit.radius * EDGE_ANCHOR_FRAC)
    return wall_gap(fruit, sign) <= limit


def _top_crown(fruits: list[Fruit]) -> float:
    """The topmost crown y. The floor if empty."""
    if not fruits:
        return float(NORMALIZED_HEIGHT)
    return min(f.y - f.radius for f in fruits)


def _height_variance(fruits: list[Fruit]) -> float:
    """Spread of crowns per column bin. 0 if empty."""
    flat_bin = 40.0
    bins: dict[int, float] = {}
    for fruit in fruits:
        key = int(fruit.x // flat_bin)
        top = fruit.y - fruit.radius
        bins[key] = min(bins.get(key, float(NORMALIZED_HEIGHT)), top)
    if len(bins) < 2:
        return 0.0
    return float(statistics.pstdev(list(bins.values())))


def _floor_row(fruits: list[Fruit] | tuple[Fruit, ...]) -> list[Fruit]:
    """Fruits on the floor in x order."""
    return sorted(
        (f for f in fruits if f.y > NORMALIZED_HEIGHT - f.radius * FLOOR_BAND),
        key=lambda f: f.x,
    )


def _floor_gap(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """The widest gap on the floor. Gaps to the walls count too. The board width if empty."""
    row = _floor_row(fruits)
    if not row:
        return float(NORMALIZED_WIDTH)
    worst = max(
        row[0].x - row[0].radius,
        NORMALIZED_WIDTH - (row[-1].x + row[-1].radius),
    )
    for left, right in zip(row, row[1:]):
        worst = max(worst, (right.x - right.radius) - (left.x + left.radius))
    return worst


def _floor_packed(fruits: list[Fruit] | tuple[Fruit, ...]) -> bool:
    """Whether the floor is filled.

    It need not be connected from wall to wall. If the gap is at most the orange's diameter,
    dropping there is not a problem, so it is considered filled.
    """
    return _floor_gap(fruits) <= FLOOR_PACKED_GAP


def _big_cluster_edge(
    fruits: list[Fruit] | tuple[Fruit, ...],
    max_type: int,
    sign: int,
) -> float:
    """The small-side edge of the big-fruit cluster.

    Not just the single biggest fruit, but down to 2 tiers below it as the cluster (the same grouping as
    _big_layout_penalty). Cutting at the biggest alone would treat the neighboring pear and apple as small side too.
    """
    big_min = max(0, max_type - BIG_CLUSTER_SPAN)
    bigs = [fruit for fruit in fruits if fruit.type >= big_min]
    if sign > 0:
        return max(fruit.x + fruit.radius for fruit in bigs)
    return min(fruit.x - fruit.radius for fruit in bigs)


def _widest_gap(
    fruits: list[Fruit] | tuple[Fruit, ...],
    lo: float,
    hi: float,
) -> tuple[float, float]:
    """Width and center x of the widest gap on the floor within [lo, hi]."""
    widest = 0.0
    center = (lo + hi) / 2.0
    cursor = lo
    for fruit in _floor_row(fruits):
        if fruit.x + fruit.radius <= lo or fruit.x - fruit.radius >= hi:
            continue
        gap = (fruit.x - fruit.radius) - cursor
        if gap > widest:
            widest = gap
            center = (cursor + (fruit.x - fruit.radius)) / 2.0
        cursor = max(cursor, fruit.x + fruit.radius)
    gap = hi - cursor
    if gap > widest:
        widest = gap
        center = (cursor + hi) / 2.0
    return widest, center


def _small_side_room_ok(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    held_r: float,
    max_type: int,
    sign: int,
) -> bool:
    """Actually drop on the small side and check whether it lands on the floor.

    Measuring floor gaps by geometry alone misses cases where another fruit overhangs the gap and it
    does not actually fit (fixed after it was pointed out). Only when the width looks physically enough,
    actually drop once into the center of the widest gap to check.
    """
    edge = _big_cluster_edge(fruits, max_type, sign)
    lo, hi = (edge, float(NORMALIZED_WIDTH)) if sign > 0 else (0.0, edge)
    lo, hi = max(lo, held_r), min(hi, NORMALIZED_WIDTH - held_r)
    if lo > hi:
        return False
    widest, center = _widest_gap(fruits, lo, hi)
    if widest < held_r * 2.0:
        return False
    x = clamp_drop_x(center, drop_type)
    after, merges, _merge_types, held_merged = simulate_drop_held(fruits, drop_type, x)
    if merges > 0:
        return True
    land_x, land_y = landed_xy(fruits, after, drop_type, x, held_r, held_merged)
    floor_y = NORMALIZED_HEIGHT - held_r
    return land_y >= floor_y - 4.0 and lo - MERGE_SLACK <= land_x <= hi + MERGE_SLACK


def _straight_fall_contact(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
) -> Fruit | None:
    """The first fruit touched when dropping straight down column x. None if only the floor.

    Not the actual landing after bouncing and rolling, but the geometric first contact
    when falling straight down the aimed column (no physics is run).
    """
    best: Fruit | None = None
    best_y = float(NORMALIZED_HEIGHT) - held_r  # the floor if it touches nothing.
    for fruit in fruits:
        dx = abs(fruit.x - x)
        gap = fruit.radius + held_r
        if dx >= gap:
            continue
        touch_y = fruit.y - math.sqrt(gap * gap - dx * dx)
        if touch_y < best_y:
            best_y = touch_y
            best = fruit
    return best


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


def _is_nestled(
    fruit: Fruit,
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> bool:
    """Whether it sits in a valley between bigger fruits."""
    return _valley_flanks(fruits, fruit.x, fruit.type) is not None


def _size_order_exempt(
    fruit: Fruit,
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> bool:
    """Whether a fruit is excluded from size order.

    Being in a valley alone does not exclude it. A fruit with no partner has no prospect of leaving that valley,
    and stays as a plain ordering violation. Excluding it would count the fruit that created the inversion as a valley wall,
    a loophole that exempts the inversion it created itself (move 9 of
    seed=49140: on a pear and grape board, placing a dekopon to the right of the grape makes that dekopon
    form a valley, dropping the grape's inversion from 1.5 to 0.14, and it beats the reordering move by 0.20
    ).

    Valleys of held's type cannot be seen from here, but those are picked up by the per-move `valley_grow_ok`
    with `VALLEY_GROW_BONUS`, so it does not crush growing.
    """
    if not _is_nestled(fruit, fruits):
        return False
    return any(f.type == fruit.type and f is not fruit for f in fruits)


def valley_grow_ok(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    drop_type: int,
    next_type: int | None,
) -> bool:
    """Whether it is a move that goes to grow a valley. The reference is the fruit in the valley; the wall types are not looked at.

    Targeting the valley fruit (a fruit squeezed between bigger fruits),
    - the fruit is the same type as held → dropping merges immediately
    - the fruit is one bigger than held, and held and next are the same type → dropping two
      merges into that fruit's type, which merges again

    Called on the pre-drop board (`before`). held is not on the board yet, so
    it never mixes itself into the target fruits.
    """
    for fruit in fruits:
        if fruit.type == drop_type:
            pass
        elif (
            next_type is not None
            and drop_type == next_type
            and fruit.type == drop_type + 1
        ):
            pass
        else:
            continue
        # The valley is taken with the target fruit as the reference. With held as the reference, the target fruit itself
        # ends up on the wall side as 'a bigger fruit' (the grape in a valley seen from a strawberry).
        flanks = _valley_flanks(fruits, fruit.x, fruit.type)
        if flanks is None:
            continue
        left, right = flanks
        if left.x < land_x < right.x:
            return True
    return False


# --- Penalty terms ---------------------------------------------------------------


def board_penalties(
    fruits: list[Fruit], *, sign: int = 1, exempt_size_order: bool = False
) -> float:
    """Board penalties after the drop (danger, burying, excess same type, size order, pushing big, bumpiness).

    exempt_size_order: True when held merged this move. Unrelated fruits knocked by the merge recoil
    are not penalized as size-order violations (see `policy._evaluate_drop`).
    """
    # Converted from the old basis of 90.0 by the amount the board moved to the inside-of-the-wall basis.
    danger_y = 70.9
    danger_crown_weight = 0.5
    bury_weight = 20.0
    variance_weight = 0.08
    variance_danger_scale = 0.15

    penalty = 0.0
    crown = _top_crown(fruits)
    if crown < danger_y:
        penalty += (danger_y - crown) * danger_crown_weight

    penalty += bury_weight * _bury_penalty(fruits)
    penalty += _excess_same_penalty(fruits)
    if not exempt_size_order:
        penalty += _size_order_penalty(fruits, sign)
    penalty += _big_layout_penalty(fruits, sign)
    variance = _height_variance(fruits)
    if crown < danger_y:
        variance *= variance_danger_scale
    penalty += variance_weight * variance
    return penalty


def _big_layout_penalty(fruits: list[Fruit] | tuple[Fruit, ...], sign: int = 1) -> float:
    """Proximity between big fruits, and the corner pocket penalty at the big-side edge.

    sign=+1 means the left is the big side, -1 the right. The corner pocket looks only at that side.
    When the biggest fruit L is on the big-side wall, small fruits outside L and below L.y are heavily penalized.
    """
    if not fruits:
        return 0.0
    max_t = max(fruit.type for fruit in fruits)

    cluster_weight = 0.025
    under_l_weight = 50.0
    big_min = max(0, max_t - BIG_CLUSTER_SPAN)
    large_left = sign > 0

    penalty = 0.0
    max_fruits = [fruit for fruit in fruits if fruit.type == max_t]

    for big in max_fruits:
        if not is_wall_anchored(big, sign):
            continue
        for fruit in fruits:
            if fruit.type >= max_t:
                continue
            if fruit.y <= big.y:
                continue
            on_outer = fruit.x < big.x if large_left else fruit.x > big.x
            if not on_outer:
                continue
            depth = fruit.y - big.y
            penalty += under_l_weight * (1.0 + 0.05 * (max_t - fruit.type))
            penalty += 0.15 * depth

    bigs = sorted(
        (fruit for fruit in fruits if fruit.type >= big_min),
        key=lambda fruit: fruit.x,
    )
    for i in range(len(bigs) - 1):
        left, right = bigs[i], bigs[i + 1]
        gap = (right.x - left.x) - left.radius - right.radius
        if gap <= 0:
            continue
        # Keep open the place for growing the tier in between. Closing it leaves no place when the type
        # that fits between is drawn later, and the only option is to send it outside and break the order
        # (measured: pulling a grape right beside an opening orange makes the next dekopon
        # fall outside the grape, giving the order 4-2-3). What is kept open is only 'one fruit needed
        # next' = the diameter of the largest missing type, and
        # any excess beyond that is penalized as before.
        missing = range(min(left.type, right.type) + 1, max(left.type, right.type))
        want = 2.0 * max((fruit_radius(t) for t in missing), default=0.0)
        gap -= want
        if gap <= 0:
            continue
        gap = min(gap, left.radius + right.radius)
        size = 0.5 + 0.05 * (left.type + right.type)
        penalty += cluster_weight * gap * size
    return penalty


def _excess_same_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """Penalize the excess when there are 3 or more of the same type. Up to 2 are allowed as waiting to merge.

    Making it triangular (the excess counting faster than linearly) was tried, but a 40-episode paired comparison
    (same seeds, baseline 2143.57 -> 2079.05) showed no significant improvement. The cause of death traced by measurement
    (low tiers lose merge partners and scatter → no retreat left late in the game) is
    not a matter of penalty weights but of layouts where merge candidates get squeezed from the side by big fruits of other types
    and become physically unmergeable. If anything is done, it belongs in candidate selection, not here.
    """
    excess_same_weight = 20.0
    counts: dict[int, int] = {}
    for fruit in fruits:
        counts[fruit.type] = counts.get(fruit.type, 0) + 1
    penalty = 0.0
    for count in counts.values():
        if count >= 3:
            penalty += (count - 2) * excess_same_weight
    return penalty


def _size_order_penalty(fruits: list[Fruit], sign: int = 1) -> float:
    """Penalize pairs whose left-right size order is inverted. Looks at relative order rather than absolute ideal positions.

    Only fruits stuck in a valley of bigger fruits and with a same-type partner left on the board
    are excluded from size order (so layout penalties do not crush valley growing). The condition is `_size_order_exempt`.
    """
    if not fruits:
        return 0.0
    size_order_pair_weight = 1.5
    size_order_ideal_weight = 0.004
    penalty = 0.0
    # _size_order_exempt is O(n) per fruit. Recomputing it per pair makes it O(n^3),
    # so compute it once up front. This runs for every candidate.
    exempt = [_size_order_exempt(f, fruits) for f in fruits]
    open_fruits = [f for f, skip in zip(fruits, exempt) if not skip]
    for i, a in enumerate(fruits):
        if exempt[i]:
            continue
        for j in range(i + 1, len(fruits)):
            b = fruits[j]
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            if exempt[j]:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            if sign > 0 and left.type < right.type:
                penalty += (right.type - left.type) * size_order_pair_weight
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * size_order_pair_weight
    if open_fruits:
        penalty += (
            sum(abs(f.x - ideal_x(f.type, sign)) for f in open_fruits)
            / len(open_fruits)
            * size_order_ideal_weight
        )
    return penalty


def _bury_penalty(fruits: list[Fruit]) -> float:
    """How much merge candidates are buried by other types."""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type <= under.type:
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


def bury_block_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
) -> float:
    """Penalty for blocking a fruit waiting for a same-type pair with a bigger fruit of another type, from directly above or the shoulder."""
    bury_block_weight = 14.0
    bury_shoulder_scale = 0.5
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
        scale = 1.0 if dx <= under.radius * 0.5 else bury_shoulder_scale
        penalty += scale * bury_block_weight * (drop_type - under.type)
    return penalty


def foreign_aim_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    held_r: float,
) -> float:
    """Penalty for whether the aimed column x is directly above a different type.

    0 if the fruit below is the same type (waiting to merge). Shoulder and floor landings are 0 too.
    """
    under = _straight_fall_contact(fruits, x, held_r)
    if under is None or under.type == drop_type:
        return 0.0
    # If the center is off, it is not directly above. Shoulder landings are out of scope.
    if abs(x - under.x) > under.radius * FOREIGN_AIM_CENTER_FRAC:
        return 0.0
    return FOREIGN_AIM_PENALTY


def packed_small_side_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    drop_type: int,
    held_r: float,
    sign: int,
) -> float:
    """Penalty for escaping a larger draw to the small side after the floor fills.

    Once the floor fills there is no side-by-side place on the small side. Placing there crushes the small fruits below and collapses.
    Penalize landing on the small side of the biggest fruit's inner edge, choosing moves that put it on the big side's shoulder.
    Not applied to merging moves (the caller calls it only when merges == 0).
    """
    if drop_type < PACKED_BIG_DRAW_MIN_TYPE:
        return 0.0
    if not fruits or not _floor_packed(fruits):
        return 0.0
    max_type = max(fruit.type for fruit in fruits)
    # If there are only fruits the same size or smaller, the notion of a big side does not stand.
    if max_type <= drop_type:
        return 0.0
    if (land_x - _big_cluster_edge(fruits, max_type, sign)) * sign <= 0.0:
        return 0.0
    # If this draw can be placed cleanly on the small side, placing it there is the normal move.
    # Send it to the big side only when 'there is no choice'. Cutting on a uniform gap width
    # fires all through the midgame and dries up the small side's production line (orange->apple->pear).
    if _small_side_room_ok(fruits, drop_type, held_r, max_type, sign):
        return 0.0
    return PACKED_SMALL_SIDE_WEIGHT
