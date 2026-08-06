"""Decide the drop column. A thin bootstrap policy (the groundwork for RL).

It has no concrete procedures (push-ins, restoring pushes, cascade gap opening, ladder firing and the like).
It only looks at merging, dangerous height, burying, light size order and rolling accident prevention.
Big fruits stay close, and the corner pocket at the big-side edge (below L's center) is avoided.
Valley growing is limited to waiting for a same type, or held/next both one smaller than the walls.
Moves are scored as eval = score (the real game's merge points) - penalties (penalties for accidents and bad moves).
"""

from __future__ import annotations

import itertools
import math
import os
import statistics
from concurrent.futures import Executor

from .observe import Observation, clamp_drop_x
from .reward import merge_score
from .sim_physics import landed_xy
from .sim_physics import simulate_drop_held
from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE, SPAWN_MAX_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# --- Tuning shared across several places ---
# Tolerance for a contact that could merge (difference between center distance and sum of radii).
MERGE_SLACK = 18.0
# Discount for the next move.
NEXT_DISCOUNT = 0.55
# Sideways offset of a landing counted as directly above a different type (ratio to the lower fruit's radius).
FOREIGN_AIM_CENTER_FRAC = 0.20
# Penalty for landing in the center band of a different type directly below.
FOREIGN_AIM_PENALTY = 100.0
# Search coarseness. The physics (simulate_drop) dominates, and this nearly decides the run time.
# The old 8/16 took 3.8 seconds per move and collection could not keep up. Traded for 1.2 seconds / score -3.4%.
# Number of held candidates that get the next lookahead. The physics is heavy, so only the top.
HELD_TOP = 2
# Candidate spacing of the next lookahead. Coarser than held (CANDIDATE_STEP).
NEXT_CANDIDATE_STEP = 32.0
# Uniform spacing of held candidates. Coarser puts the spot directly above a dangerous pile among the candidates, so do not raise it
# (test_avoids_dangerous_tall_stack failed at 20). Speed is earned on the lookahead side.
CANDIDATE_STEP = 12.0

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

# --- Ladder detection (the shape that fires a corner big fruit up a staircase) ---
# Put a peach (7) in the corner, a pear (6) next to it on the inside. On the shoulders of those two, an apple (5) and an orange (4).
# Dropping an orange last cascades 4→5→6→7 and the corner peach becomes a pineapple.
# The same holds from a corner pineapple onward; the staircase always goes down to the biggest drawable type (orange).
#
# For now it only detects and is not used for move selection. What measurement has shown:
# - Firing needs no guidance. Once a ladder is built, choose_x ties with the best of an exhaustive sweep over x
# - Boards where it gets built do not appear (4 full rungs 12 times in 720 measured boards). This is where to intervene
# - Without a filled floor the shape does not hold. The pear is pushed out like a wedge and self-destructs,
#   and wherever you drop you get only one rung (15 points). A filled floor is a gate condition
LADDER_MIN_ANCHOR_TYPE = 7
# The bottom rung of the ladder.
LADDER_BASE_TYPE = SPAWN_MAX_TYPE
# The one below (dekopon). Two dekopons can make the orange rung, so
# it counts as a rung only when held/next are both dekopon.
LADDER_FEED_TYPE = SPAWN_MAX_TYPE - 1

# --- Big draws after the floor fills ---
# When the floor fills there is no place left on the small side. Still _ideal_x keeps pulling small fruits
# to the small side (orange's ideal is 236 = right side), so larger draws get stacked
# on the small side, crushing the small fruits below and collapsing. Once the floor fills, put them on the big side's shoulder
# instead of side by side. The ladder shape comes out as a result of this placement split.
# Measured (10 seeds × 120 moves): of 358 cases, 211 were placed on the small side, and in 210 of them
# eval really chose the small side (median +4.1). A problem of evaluation, not candidates.
PACKED_BIG_DRAW_MIN_TYPE = SPAWN_MAX_TYPE - 1
# It flips the narrow median margin of +4.1 while keeping moves that can actually merge on the small side (max +159.9).
# It is not applied to merging moves (only when merges == 0), so it does not compete with merging.
PACKED_SMALL_SIDE_WEIGHT = 8.0

# Switch for staged rollout. 0 returns placement after the floor fills to the old behavior (for A/B).
PACKED_RULE_ENABLED = os.environ.get("SUIKA_PACKED", "1") != "0"


def set_packed_rule_enabled(enabled: bool) -> None:
    """Enable/disable placement after the floor fills (for A/B comparison and tests)."""
    global PACKED_RULE_ENABLED
    PACKED_RULE_ENABLED = enabled


def _held_eval_job(
    obs: Observation, held_r: float, x: float
) -> tuple[float, float, list[Fruit]]:
    """(eval, x, after) for one held candidate. The unit sent to the pool."""
    after, held_eval = _held_eval(obs, x, held_r)
    return held_eval, x, after


def choose_x(obs: Observation, *, pool: Executor | None = None) -> float:
    """Return the column to drop from the observation. Assumes ready with held_type present.

    Passing pool spreads the simulate_drop of held/next candidates over a process pool.
    The result is the same as serial execution (every candidate is independent and the board is only read).
    """
    if obs.held_type is None:
        raise ValueError("no held_type")

    held_r = fruit_radius(obs.held_type)
    xs = [
        clamp_drop_x(x, obs.held_type)
        for x in _candidates(list(obs.fruits), obs.held_type, held_r, extra_type=obs.next_type)
    ]
    if not xs:
        return NORMALIZED_WIDTH / 2

    if pool is None:
        ranked = [_held_eval_job(obs, held_r, x) for x in xs]
    else:
        ranked = list(
            pool.map(_held_eval_job, itertools.repeat(obs), itertools.repeat(held_r), xs)
        )

    ranked.sort(key=lambda row: row[0], reverse=True)
    if obs.next_type is None:
        return ranked[0][1]

    # The next lookahead covers only the top held eval (the physics is heavy). Candidates are coarser than held.
    best_x = ranked[0][1]
    best_score = -math.inf
    for held_eval, x, after in ranked[:HELD_TOP]:
        value = held_eval + NEXT_DISCOUNT * _best_next_score(
            after, obs.next_type, step=NEXT_CANDIDATE_STEP, pool=pool
        )
        if value > best_score:
            best_score = value
            best_x = x
    return best_x


def _candidates(
    fruits: tuple[Fruit, ...] | list[Fruit],
    drop_type: int,
    held_r: float,
    extra_type: int | None = None,
    *,
    step: float | None = None,
) -> list[float]:
    """Uniform spacing plus spots above / beside same-type and nearby fruits, and ideal_x."""
    sign = _order_sign(fruits)
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    grid = CANDIDATE_STEP if step is None else step
    # List every multiple of grid within lo..hi.
    xs = {i * grid for i in range(math.ceil(lo / grid), int(hi / grid) + 1)}
    xs.add(_ideal_x(drop_type, sign))
    _add_near_fruit_x(xs, fruits, held_r, lambda t: drop_type <= t <= drop_type + 2)

    if extra_type is not None:
        xs.add(_ideal_x(extra_type, sign))
        _add_near_fruit_x(xs, fruits, held_r, lambda t: t == extra_type)

    return [x for x in xs if lo <= x <= hi]


def _add_near_fruit_x(
    xs: set[float],
    fruits: tuple[Fruit, ...] | list[Fruit],
    held_r: float,
    matches,
) -> None:
    """Add the positions above / touching left and right of fruits whose type satisfies matches to xs."""
    for fruit in fruits:
        if not matches(fruit.type):
            continue
        xs.add(fruit.x)
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)


def drop_scores(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    *,
    next_type: int | None = None,
) -> tuple[float, float, float, list[Fruit], int]:
    """(score, penalties, eval, after, merges) of one move dropped at column x.

    For sim / training. after and merges are the simulate_drop results as is.
    Board penalties are the difference from before the drop. On the same board it is a constant difference, so choose_x's
    choice does not change, and summing it per move does not grow with the size of the board.
    """
    held_r = fruit_radius(drop_type)
    before = list(fruits)
    after, score, penalties, merges, held_merged = _evaluate_drop(
        before,
        drop_type,
        clamp_drop_x(x, drop_type),
        held_r,
        next_type=next_type,
    )
    # Subtract the pre-drop part on the same basis as the post-drop part. Excluding size_order on the after side
    # while subtracting it included on the before side subtracts a penalty that does not exist, and merge moves
    # become unfairly favored (a measured offset of 0.303).
    penalties -= _board_penalties(
        before, sign=_order_sign(before), exempt_size_order=held_merged
    )
    return score, penalties, score - penalties, after, merges


def _held_eval(obs: Observation, x: float, held_r: float) -> tuple[list[Fruit], float]:
    """(board, score - penalties) after dropping held at x. Does not look at next."""
    assert obs.held_type is not None
    before = list(obs.fruits)
    after, score, penalties, _merges, _held_merged = _evaluate_drop(
        before, obs.held_type, x, held_r, next_type=obs.next_type
    )
    return after, score - penalties


def _score(obs: Observation, x: float, held_r: float) -> float:
    """Score the board after dropping held + the hypothetical best move of next."""
    after, value = _held_eval(obs, x, held_r)
    if obs.next_type is not None:
        value += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)
    return value


def _next_eval_job(fruits: list[Fruit], next_type: int, next_r: float, nx: float) -> float:
    """eval for one next candidate. The unit sent to the pool."""
    # The next after that is unknown. Only same-type fruit in a valley counts for the growing exemption.
    _, score, penalties, _merges, _held_merged = _evaluate_drop(
        fruits, next_type, nx, next_r
    )
    return score - penalties


def _best_next_score(
    fruits: list[Fruit],
    next_type: int,
    *,
    step: float | None = None,
    pool: Executor | None = None,
) -> float:
    """eval when next is dropped at its best column."""
    next_r = fruit_radius(next_type)
    xs = [clamp_drop_x(nx, next_type) for nx in _candidates(fruits, next_type, next_r, step=step)]
    if not xs:
        return 0.0
    if pool is None:
        scores = [_next_eval_job(fruits, next_type, next_r, nx) for nx in xs]
    else:
        scores = list(
            pool.map(
                _next_eval_job,
                itertools.repeat(fruits),
                itertools.repeat(next_type),
                itertools.repeat(next_r),
                xs,
            )
        )
    return max(scores)


def _evaluate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    held_r: float,
    *,
    next_type: int | None = None,
) -> tuple[list[Fruit], float, float, int, bool]:
    """Board, real-game score, penalties, merge count and whether held merged after one drop."""
    before = list(fruits)
    sign = _order_sign(before)
    after, merges, merge_types, held_merged = simulate_drop_held(before, drop_type, x)
    land_x, land_y = landed_xy(before, after, drop_type, x, held_r, held_merged)

    score = merge_score(merge_types)
    # When held (this move) merged, size-order and burying penalties are not applied to unrelated fruits
    # knocked by its recoil. Looking only at `merges >= 1` would also exempt merges that happened by chance elsewhere
    # on the board unrelated to held,
    # so judge by whether held itself took part in a merge (`held_merged`).
    penalties = _board_penalties(after, sign=sign, exempt_size_order=held_merged)
    # FOREIGN_AIM looks at 'is the fruit directly below a different type', not merges.
    # A same type directly below is OK (waiting to merge). Rolling off a different type and merging on the floor is still penalized.
    penalties += _foreign_aim_penalty(before, x, drop_type, held_r)
    if not held_merged:
        penalties += _bury_block_penalty(before, land_x, land_y, drop_type, held_r)
        penalties += _packed_small_side_penalty(before, land_x, drop_type, held_r, sign)
    return after, score, penalties, merges, held_merged


def _board_penalties(
    fruits: list[Fruit], *, sign: int = 1, exempt_size_order: bool = False
) -> float:
    """Board penalties after the drop (danger, burying, excess same type, size order, pushing big, bumpiness).

    exempt_size_order: True when held merged this move. Unrelated fruits knocked by the merge recoil
    unrelated fruits knocked away are not penalized as size-order violations (see `_evaluate_drop`).
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
        if not _is_wall_anchored(big, sign):
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
        gap = min(gap, left.radius + right.radius)
        size = 0.5 + 0.05 * (left.type + right.type)
        penalty += cluster_weight * gap * size
    return penalty


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


def _wall_gap(fruit: Fruit, sign: int) -> float:
    """Gap to the wall on the big side (sign)."""
    if sign > 0:
        return fruit.x - fruit.radius
    return NORMALIZED_WIDTH - fruit.radius - fruit.x


def _is_wall_anchored(fruit: Fruit, sign: int) -> bool:
    """Whether it is on the big-side wall."""
    limit = max(EDGE_ANCHOR_MIN, fruit.radius * EDGE_ANCHOR_FRAC)
    return _wall_gap(fruit, sign) <= limit


def _ladder_anchor(
    fruits: list[Fruit] | tuple[Fruit, ...],
    sign: int,
) -> Fruit | None:
    """The base of the ladder. The biggest fruit on the big-side wall. None if below peach or away from the wall."""
    if not fruits:
        return None
    max_t = max(fruit.type for fruit in fruits)
    if max_t < LADDER_MIN_ANCHOR_TYPE:
        return None
    best: Fruit | None = None
    for fruit in fruits:
        if fruit.type != max_t or not _is_wall_anchored(fruit, sign):
            continue
        if best is None or _wall_gap(fruit, sign) < _wall_gap(best, sign):
            best = fruit
    return best


def _ladder_window(anchor: Fruit, sign: int) -> tuple[float, float]:
    """The horizontal band the ladder occupies. From slightly outside the base's center, to two pears' worth on the inside."""
    inner = anchor.radius + fruit_radius(anchor.type - 1) * 2.0 + MERGE_SLACK
    outer = anchor.radius * 0.5
    if sign > 0:
        return anchor.x - outer, anchor.x + inner
    return anchor.x - inner, anchor.x + outer


def _ladder_beside_anchor(anchor: Fruit, x: float, sign: int) -> bool:
    """Whether it is next to the base on the inside, not directly on top.

    The rung one smaller (the pear for a peach) goes alongside. Stacking it directly on top makes a shape that collapses.
    """
    return (x - anchor.x) * sign > anchor.radius * 0.5


def _ladder_rungs(
    fruits: list[Fruit] | tuple[Fruit, ...],
    anchor: Fruit,
    sign: int,
) -> dict[int, Fruit]:
    """Rungs filled continuously downward from the base. Ends where it breaks.

    Rungs are inside the horizontal band, taken from the wall side, and not lower than the rung above.
    The pear is next to the peach (about the same y), so the floor radius difference is allowed.
    """
    lo, hi = _ladder_window(anchor, sign)
    rungs = {anchor.type: anchor}
    above = anchor
    for want in range(anchor.type - 1, LADDER_FEED_TYPE - 1, -1):
        best: Fruit | None = None
        for fruit in fruits:
            if fruit.type != want or fruit is anchor:
                continue
            if not lo <= fruit.x <= hi:
                continue
            if fruit.y > above.y + above.radius:
                continue
            if want == anchor.type - 1 and not _ladder_beside_anchor(
                anchor, fruit.x, sign
            ):
                continue
            if best is None or _wall_gap(fruit, sign) < _wall_gap(best, sign):
                best = fruit
        if best is None:
            break
        rungs[want] = best
        above = best
    return rungs


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


def _foreign_aim_penalty(
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


def _packed_small_side_penalty(
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
    if not PACKED_RULE_ENABLED or drop_type < PACKED_BIG_DRAW_MIN_TYPE:
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
    size_order_pair_weight = 1.5
    size_order_ideal_weight = 0.004
    penalty = 0.0
    # _is_nestled is O(n) per fruit. Recomputing it per pair makes it O(n^3), so
    # compute it once up front. This runs for every candidate.
    nestled = [_is_nestled(f, fruits) for f in fruits]
    open_fruits = [f for f, nest in zip(fruits, nestled) if not nest]
    for i, a in enumerate(fruits):
        if nestled[i]:
            continue
        for j in range(i + 1, len(fruits)):
            b = fruits[j]
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            if nestled[j]:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            if sign > 0 and left.type < right.type:
                penalty += (right.type - left.type) * size_order_pair_weight
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * size_order_pair_weight
    if open_fruits:
        penalty += (
            sum(abs(f.x - _ideal_x(f.type, sign)) for f in open_fruits)
            / len(open_fruits)
            * size_order_ideal_weight
        )
    return penalty


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


def _bury_block_penalty(
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


