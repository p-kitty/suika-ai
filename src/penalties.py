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

from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# --- Tuning shared across several places ---
# Tolerance for a contact that could merge (difference between center distance and sum of radii).
MERGE_SLACK = 18.0
# Sideways offset of a landing counted as directly above a different type (ratio to the lower fruit's radius).
FOREIGN_AIM_CENTER_FRAC = 0.20
# Penalty for landing in the center band of a different type directly below.
FOREIGN_AIM_WEIGHT = 100.0
# A weight only for ordering tied candidates. Positions where every other term ties are the norm, and without it
# the winning move would be decided by an implementation detail: the enumeration order of the candidate set = float hash order.
# The smallest merge score is 1.0 (cherry -> straw), so to avoid overturning real differences it is kept at most
# a fifth of that (`|x - center|` is at most 190, so 0.001 gives 0.19).
# Do not try to express how good a move is with this.
CENTER_TIEBREAK_WEIGHT = 0.001

# Bonus for merges pushed to the big side (applied by subtracting from penalties). The value `merge_big_side_bonus`
# returns. Does not break the property that merges are ranked by the real game's score. The smallest
# merge score difference is 1.0 (cherry -> straw), so it is kept to a value that does not overturn it.
# 5x the tie band width of 0.1.
MERGE_BIG_SIDE_WEIGHT = 0.5
# The minimum movement counted as pushed (ratio to the dropped fruit's radius). The merge position is the midpoint
# of the two centers, so even moves with no intent to push normally shift by less than a radius.
MERGE_BIG_SIDE_SLACK_FRAC = 1.0

# Valley-growing bonus (applied by subtracting from penalties). The value `valley_grow_bonus` returns.
# Not stronger than a real merge. At 8.0 it rejected a grape merge (6 points) for a non-merging valley.
# At 2.0 it tips toward growing, and at 3.0 it still keeps taking merges (measured).
VALLEY_GROW_WEIGHT = 3.0

# --- Wall-anchored check (shared by the corner pocket penalty and the ladder base) ---
EDGE_ANCHOR_MIN = 24.0
EDGE_ANCHOR_FRAC = 0.35

# --- Board penalty weights ---
# The A/B in compare_policy swaps them as module attributes, so
# they live here rather than as locals of board_penalties.
# Per buried fruit. A fruit with a partner on the board could have merged next move, so it is heavy.
BURY_WEIGHT = 20.0
# Per buried fruit with no partner on the board. The partner is yet to be drawn, and roofing it
# leaves that fruit unable to meet anyone (46.6% of fossils have their top blocked.
# NOTES 'Measuring fruits that never merge (fossils)'). Lighter than crushing a waiting pair, but not ignored.
BURY_LONE_WEIGHT = 15.0
# Upper limit of the type gap allowed on a big fruit's shoulder. Up to an orange (4) on a pineapple's (8) shoulder is allowed.
PERCH_MIN_GAP = 5
# Range of fruits whose shoulders are checked (how many tiers below the biggest). 0 means only the biggest.
PERCH_BIG_SPAN = 1
PERCH_WEIGHT = 16.0
# Depth of a hollow that exempts a perch. If the type gap to the wall (the smaller of the hollow's left and right)
# is at most this, it is the next rung. When a partner comes it merges on the spot and can then merge with the wall.
# Measured over every candidate on 3 seeds, only 6.8% of perches are exempt. Exempting any hollow
# would remove 77.5%, and the cherry in the very position that motivated this rule
# sits in the hollow between apple and orange, so it would exempt the case itself.
PERCH_RUNG_MAX_GAP = 1
# Type gap to the wall for counting a fruit sunk in a pit. A wall one tier up is the next rung and not counted
# (same criterion as `_is_rung`).
PIT_MIN_GAP = 2
# Per pit tier. Lighter than `_perch_penalty`. A fruit on a shoulder can meet a partner coming from above,
# but a pit only has high walls with the top open, so it is recoverable.
PIT_WEIGHT = 8.0
# Per fruit from the third of a type onward.
EXCESS_SAME_WEIGHT = 20.0
# Per tier of left-right size inversion.
SIZE_ORDER_PAIR_WEIGHT = 1.5
# Multiplies the mean deviation from ideal_x.
SIZE_ORDER_IDEAL_WEIGHT = 0.004



# --- Helpers that only read the board -------------------------------------------------


def ideal_x(fruit_type: int, sign: int = 1) -> float:
    """With sign=+1, bigger goes left. With sign=-1, bigger goes right."""
    base = NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))
    if sign < 0:
        return NORMALIZED_WIDTH - base
    return base


def center_tiebreak(x: float) -> float:
    """A center-leaning penalty only for breaking ties. Edge moves drop out first."""
    return CENTER_TIEBREAK_WEIGHT * abs(x - NORMALIZED_WIDTH / 2)


def wall_gap(fruit: Fruit, sign: int) -> float:
    """Gap to the wall on the big side (sign)."""
    if sign > 0:
        return fruit.x - fruit.radius
    return NORMALIZED_WIDTH - fruit.radius - fruit.x


def is_wall_anchored(fruit: Fruit, sign: int) -> bool:
    """Whether it is on the big-side wall."""
    limit = max(EDGE_ANCHOR_MIN, fruit.radius * EDGE_ANCHOR_FRAC)
    return wall_gap(fruit, sign) <= limit


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

    **The partner must be in the same valley.** If one anywhere on the board were enough,
    a partner outside the valley would grant the exemption without being able to reach. The valley is made of fruits
    bigger than itself, so it cannot merge with a partner on the other side, and the fruit just stays
    breaking the order (move 35 of seed=834761: a strawberry left in the valley of a pear and a pineapple
    was exempted on the basis of a strawberry at the opposite edge x=23, and the inversion of 7.5 became 0).

    Valleys of held's type cannot be seen from here, but those are picked up by the per-move `valley_grow_bonus`,
    so it does not crush growing.
    """
    flanks = _valley_flanks(fruits, fruit.x, fruit.type)
    if flanks is None:
        return False
    left, right = flanks
    return any(
        f.type == fruit.type and f is not fruit and left.x < f.x < right.x
        for f in fruits
    )


def valley_grow_bonus(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    drop_type: int,
    next_type: int | None,
) -> float:
    """A bonus for moves that go to grow a valley. The reference is the fruit in the valley; the wall types are not looked at.

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
            return VALLEY_GROW_WEIGHT
    return 0.0


def merge_big_side_bonus(
    drop_x: float, held_fruit: Fruit | None, held_r: float, sign: int
) -> float:
    """A bonus for moves where the fruit made by the merge ends up at least one radius toward the big side of the drop column.

    `held_fruit` is where the dropped fruit's lineage ends up (`simulate_drop_held`). A merge
    throws the new fruit sideways by recoil, so hitting the same partner from the left or right
    changes the resulting layout. Including cases that hit after rolling, it makes the policy choose
    the way of hitting that ends up toward the big side (`sign`).

    Applied only to merging moves. Landings of non-merging moves are seen directly by `_size_order_penalty`,
    so they are not looked at twice here.
    """
    if held_fruit is None:
        return 0.0
    toward_big = -sign * (held_fruit.x - drop_x)
    if toward_big < MERGE_BIG_SIDE_SLACK_FRAC * held_r:
        return 0.0
    return MERGE_BIG_SIDE_WEIGHT


# --- Penalty terms ---------------------------------------------------------------


def board_penalties(
    fruits: list[Fruit], *, sign: int = 1, exempt_size_order: bool = False
) -> float:
    """Board penalties after the drop (burying, perch, excess same type, size order, corner pocket).

    exempt_size_order: True when held merged this move. Unrelated fruits knocked by the merge recoil
    are not penalized as size-order violations (see `policy._evaluate_drop`).
    """
    penalty = 0.0
    penalty += _bury_penalty(fruits)
    penalty += PERCH_WEIGHT * _perch_penalty(fruits)
    penalty += PIT_WEIGHT * _pit_penalty(fruits)
    penalty += _excess_same_penalty(fruits)
    if not exempt_size_order:
        penalty += _size_order_penalty(fruits, sign)
    penalty += _corner_pocket_penalty(fruits, sign)
    return penalty


def _corner_pocket_penalty(fruits: list[Fruit] | tuple[Fruit, ...], sign: int = 1) -> float:
    """Corner pocket penalty at the big-side edge.

    sign=+1 means the left is the big side, -1 the right. The corner pocket looks only at that side.
    When the biggest fruit L is on the big-side wall, small fruits outside L and below L.y are heavily penalized.
    A fruit that gets behind L stays without meeting its merge partner.
    """
    if not fruits:
        return 0.0
    max_t = max(fruit.type for fruit in fruits)

    under_l_weight = 50.0
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

    return penalty


def _excess_same_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """Penalize the excess when there are 3 or more of the same type. Up to 2 are allowed as waiting to merge.

    A shape making the excess count grow faster than linearly was measured and shelved
    (the improvement attempts in NOTES 'Investigated: sudden death from scattered low-tier fruits late in the game').
    """
    counts: dict[int, int] = {}
    for fruit in fruits:
        counts[fruit.type] = counts.get(fruit.type, 0) + 1
    penalty = 0.0
    for count in counts.values():
        if count >= 3:
            penalty += (count - 2) * EXCESS_SAME_WEIGHT
    return penalty


def _size_order_penalty(fruits: list[Fruit], sign: int = 1) -> float:
    """Penalize pairs whose left-right size order is inverted. Looks at relative order rather than absolute ideal positions.

    Only fruits stuck in a valley of bigger fruits and with a same-type partner left on the board
    are excluded from size order (so layout penalties do not crush valley growing). The condition is `_size_order_exempt`.
    """
    if not fruits:
        return 0.0
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
                penalty += (right.type - left.type) * SIZE_ORDER_PAIR_WEIGHT
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * SIZE_ORDER_PAIR_WEIGHT
    if open_fruits:
        penalty += (
            sum(abs(f.x - ideal_x(f.type, sign)) for f in open_fruits)
            / len(open_fruits)
            * SIZE_ORDER_IDEAL_WEIGHT
        )
    return penalty


def _bury_counts(fruits: list[Fruit] | tuple[Fruit, ...]) -> tuple[float, float]:
    """Return the number of fruits buried by other types split into (with partner, without partner).

    It is a rule with two weights, so counting and weights are separated so that `band_escape.py`
    can sweep them separately (AGENTS 'One rule per term').
    """
    paired = lone = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type <= under.type:
                continue
            if over.y >= under.y:
                continue
            # The contact window uses both radii. Based on the lower fruit alone,
            # the window narrows the more a big fruit sits on a small one, and the shape we most want to crush,
            # 'burying small with big', escapes detection.
            if abs(over.x - under.x) > (under.radius + over.radius) * 0.9:
                continue
            gap = (under.y - under.radius) - (over.y + over.radius)
            if -MERGE_SLACK <= gap <= under.radius * 0.6:
                siblings = sum(1 for f in fruits if f.type == under.type and f is not under)
                if siblings >= 1:
                    paired += 1.0
                else:
                    lone += 1.0
    return paired, lone


def _bury_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """Penalty for burying merge candidates with other types. The weight depends on whether a partner exists."""
    paired, lone = _bury_counts(fruits)
    return BURY_WEIGHT * paired + BURY_LONE_WEIGHT * lone


def _perch_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """Penalty for small fruits sitting on a big fruit's shoulders or top. Heavier with a larger type gap.

    The inverse of `_bury_penalty`. That one counts 'a different type above a small fruit', so
    it looks only at the `over.type > under.type` side, and the reverse (a small fruit on a big fruit)
    slipped through every rule. The horizontal `_size_order_penalty` also excludes vertically stacked
    pairs as the same column, and fruits stuck in valleys are removed by `_size_order_exempt`,
    so it went through untouched.

    The top of a big fruit is where the next rung is built, and putting a fruit with a large type gap there makes it
    stay without meeting a partner and also blocks the merging face of the big fruit below. Sitting on it is judged
    not by contact but by 'inside the big fruit's footprint, with its bottom above the big fruit's center'.
    Even without direct contact, it catches shapes sitting on the pile with one tier in between.

    Returns 1.0 for each step the type gap exceeds the allowed gap. The caller applies the weight.
    """
    if not fruits:
        return 0.0
    max_t = max(fruit.type for fruit in fruits)
    big_min = max_t - PERCH_BIG_SPAN
    # Whether it is a rung depends only on the fruit sitting on top, so it is not recomputed per lower fruit.
    rung: dict[int, bool] = {}
    penalty = 0.0
    for under in fruits:
        if under.type < big_min:
            continue
        for over in fruits:
            gap_type = under.type - over.type
            if gap_type < PERCH_MIN_GAP:
                continue
            if over.y + over.radius > under.y:
                continue
            if abs(over.x - under.x) > under.radius + over.radius:
                continue
            if id(over) not in rung:
                rung[id(over)] = _is_rung(over, fruits)
            if rung[id(over)]:
                continue
            penalty += float(gap_type - PERCH_MIN_GAP + 1)
    return penalty


def _pit_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """Penalty for a fruit at the bottom of a valley of much bigger fruits with no partner in the same valley.

    Where `_perch_penalty` looks 'above' a big fruit, this looks 'between'. If the walls are many tiers
    bigger than itself it cannot merge sideways, and a partner can only come through the narrow gap directly above.
    It is exactly the 'fruit with no prospect of leaving that valley' named in the docstring of `_size_order_exempt`,
    and until now its only price was the pair difference of 1.5 in `_size_order_penalty`.
    The global pair count barely moves with where one fruit goes, so that does not work
    (NOTES 'Ideas that did not work'). This is a shape the dropped fruit itself creates, so it splits between candidates.

    A type gap of 1 tier to the wall is the next rung and not counted (`PIT_MIN_GAP`). If a same-type partner
    is in the same valley it is considered growing and excluded. The check matches `_size_order_exempt`.

    Returns 1.0 for each step the type gap exceeds `PIT_MIN_GAP`. The caller applies the weight.
    """
    penalty = 0.0
    for fruit in fruits:
        flanks = _valley_flanks(fruits, fruit.x, fruit.type)
        if flanks is None:
            continue
        left, right = flanks
        gap = min(left.type, right.type) - fruit.type
        if gap < PIT_MIN_GAP:
            continue
        if any(
            f.type == fruit.type and f is not fruit and left.x < f.x < right.x
            for f in fruits
        ):
            continue
        penalty += float(gap - PIT_MIN_GAP + 1)
    return penalty


def _is_rung(fruit: Fruit, fruits: list[Fruit] | tuple[Fruit, ...]) -> bool:
    """Whether it sits in the hollow of the next rung. Distinguishes it from the bare top of a big fruit.

    When the board fills with big fruits, a small fruit **has a wide type gap on whichever shoulder it goes**. With no escape route,
    the policy avoids shoulders and tips toward roofing another small fruit (move 72 of
    seed=890270: putting a grape on a peach's shoulder costs 16, on a pineapple 32, and roofing a strawberry
    15, so the roof was cheapest). A roofed fruit cannot be reached by a partner from above,
    so something that should be heavier than a shoulder had become lighter.

    But not every hollow is fine. The cherry in the position that motivated `_perch_penalty`
    was also sitting in the hollow between apple and orange. What separates them is
    **the type gap to the wall**: with a wall one tier up (`PERCH_RUNG_MAX_GAP`), once a partner comes
    it merges and catches up with the wall.
    """
    flanks = _valley_flanks(fruits, fruit.x, fruit.type)
    if flanks is None:
        return False
    left, right = flanks
    return min(left.type, right.type) - fruit.type <= PERCH_RUNG_MAX_GAP


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
    return FOREIGN_AIM_WEIGHT

