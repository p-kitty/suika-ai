"""Unit tests for the size-order (horizontal, vertical) penalty, its valley exemption and the stage gate.

Move selection is in tests/test_policy.py. This pins down the meaning of the penalty rules themselves.
The valley check (`_is_nestled`) and the condition that actually exempts within it (`_size_order_exempt`)
are different things, so they are kept separate.

The key to vertical size order is that it 'applies only when the upper one is bigger': putting a small fruit on a big fruit's
shoulder (a ladder) must be 0 whatever the type gap. Missing that,
it goes to penalize the very moves that tidy the board.
"""

import math

from src.observe import Observation
from src.penalties import (
    VERTICAL_ORDER_WEIGHT,
    _is_nestled,
    _size_order_exempt,
    _size_order_penalty,
    _vertical_order_penalty,
    board_is_broken,
    inversion_fraction,
)
from src.policy import choose_x
from src.sim.sim_physics import simulate_drop_held
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit

PEAR, DEKOPON, GRAPE = 6, 3, 2
MELON, ORANGE, CHERRY, PEACH, APPLE = 9, 4, 0, 7, 5
# Size direction of the board. +1 = the left is big.
LARGE_LEFT = 1


def _on_floor(fruit_type: int, x: float) -> Fruit:
    radius = fruit_radius(fruit_type)
    return Fruit(
        type=fruit_type, x=x, y=NORMALIZED_HEIGHT - radius, radius=radius, confidence=90
    )


def test_nestled_needs_a_bigger_fruit_on_both_sides() -> None:
    """It counts as a valley only when squeezed on both sides by fruits bigger than itself."""
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)

    assert not _is_nestled(grape, [pear, grape])
    assert _is_nestled(grape, [pear, grape, _on_floor(DEKOPON, 230.0)])


def test_nestled_only_when_the_valley_is_narrow() -> None:
    """If the left and right are far apart it is not a valley. The same 3 fruits change verdict by spacing alone."""
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)

    assert not _is_nestled(grape, [pear, grape, _on_floor(DEKOPON, 330.0)])


def test_valley_fruit_is_exempt_only_with_a_merge_partner() -> None:
    """A fruit in a valley is excluded from size order only when a same-type partner remains on the board.

    In both boards the grape (2) is left of the dekopon (3) = inverted, and the valley shape is the same.
    The only difference is whether there is another grape to merge with. Without a partner
    there is no prospect of leaving the valley, so it counts as a plain ordering violation.
    """
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)
    dekopon = _on_floor(DEKOPON, 230.0)
    alone = [pear, grape, dekopon]
    with_partner = [pear, grape, dekopon, _on_floor(GRAPE, 300.0)]

    assert _is_nestled(grape, alone)
    assert _is_nestled(grape, with_partner)
    assert not _size_order_exempt(grape, alone)
    assert _size_order_exempt(grape, with_partner)


def test_inversion_costs_more_than_the_correct_order() -> None:
    """With the same 3 fruits, an inverted board must cost more than a correctly ordered one.

    Pear, dekopon and grape in the same positions, just swapping the middle and the right.
    On the inverted side the grape enters the valley of the pear and dekopon, so if the exemption condition
    is loose the size-order penalty falls below the correctly ordered board.
    """
    pear = _on_floor(PEAR, 70.0)
    ordered = [pear, _on_floor(DEKOPON, 170.0), _on_floor(GRAPE, 230.0)]
    inverted = [pear, _on_floor(GRAPE, 170.0), _on_floor(DEKOPON, 230.0)]

    assert _size_order_penalty(inverted, LARGE_LEFT) > _size_order_penalty(
        ordered, LARGE_LEFT
    )


def test_drop_does_not_exempt_the_inversion_it_creates() -> None:
    """Move 9 of seed=49140. Do not place the dekopon on the small side of the grape.

    On the pre-drop board the grape is not in a valley (no bigger fruit on its right).
    Placing the dekopon right of the grape forms a valley of pear and dekopon, so if the exemption condition
    is loose, the inversion that dekopon created disappears thanks to that same dekopon.
    """
    fruits = (_on_floor(PEAR, 96.5), _on_floor(GRAPE, 207.4))
    assert not _is_nestled(fruits[1], list(fruits))

    obs = Observation(
        ready=True,
        blocked=False,
        fruits=fruits,
        held_type=DEKOPON,
        held_x=NORMALIZED_WIDTH / 2,
        next_type=0,
    )
    after, _merges, _types, _held_merged = simulate_drop_held(
        list(fruits), DEKOPON, choose_x(obs)
    )
    dekopon = next(f for f in after if f.type == DEKOPON)
    grape = next(f for f in after if f.type == GRAPE)

    assert dekopon.x < grape.x


# --- Vertical size order ---------------------------------------------------------


def _stacked(lower_type: int, upper_type: int, x: float = 200.0) -> list[Fruit]:
    """Two stacked directly on top."""
    lo_r, up_r = fruit_radius(lower_type), fruit_radius(upper_type)
    lower = _on_floor(lower_type, x)
    upper = Fruit(
        type=upper_type, x=x, y=lower.y - lo_r - up_r, radius=up_r, confidence=90
    )
    return [lower, upper]


def _rest_on(a: Fruit, b: Fruit, fruit_type: int) -> Fruit:
    """A fruit resting on top touching both a and b. Scaffolding for building a shoulder shape."""
    r = fruit_radius(fruit_type)
    d1, d2 = a.radius + r, b.radius + r
    dx, dy = b.x - a.x, b.y - a.y
    d = math.hypot(dx, dy)
    mid = (d1 * d1 - d2 * d2 + d * d) / (2 * d)
    h = math.sqrt(max(0.0, d1 * d1 - mid * mid))
    cx, cy = a.x + mid * dx / d, a.y + mid * dy / d
    x1, y1 = cx + h * dy / d, cy - h * dx / d
    x2, y2 = cx - h * dy / d, cy + h * dx / d
    x, y = (x1, y1) if y1 < y2 else (x2, y2)
    return Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=90)


def test_vertical_order_ignores_a_small_fruit_on_a_big_one() -> None:
    """A small fruit on a big fruit is the correct direction. 0 even with a wide type gap.

    An orange (4) on a melon (9) is type gap 5, but correct as a vertical order.
    Counting by type gap alone here would penalize moves that tidy the board.
    """
    assert _vertical_order_penalty(_stacked(MELON, ORANGE)) == 0.0
    assert _vertical_order_penalty(_stacked(PEACH, CHERRY)) == 0.0


def test_vertical_order_charges_a_big_fruit_on_a_small_one() -> None:
    """Putting a big fruit on a small one is penalized by the type gap."""
    assert _vertical_order_penalty(_stacked(ORANGE, MELON)) == 5 * VERTICAL_ORDER_WEIGHT
    assert _vertical_order_penalty(_stacked(CHERRY, PEACH)) == 7 * VERTICAL_ORDER_WEIGHT


def test_vertical_order_ignores_fruits_merely_side_by_side() -> None:
    """Pairs merely side by side are not 'stacked'. 0 at the same height."""
    row = [_on_floor(CHERRY, 100.0), _on_floor(MELON, 300.0)]
    assert _vertical_order_penalty(row) == 0.0


def test_vertical_order_leaves_the_ladder_alone() -> None:
    """Ladder shoulders pass through. A pear inside a corner peach, and an apple and orange on the shoulders.

    This is exactly the 'wait to fire on the big side' shape, so if the vertical rule applied here
    it would crush its own building. 0 even with a type gap (3 between peach 7 and orange 4).
    """
    peach_r, pear_r = fruit_radius(PEACH), fruit_radius(PEAR)
    peach = _on_floor(PEACH, peach_r + 2)
    pear = Fruit(
        type=PEAR,
        x=peach.x + peach_r + pear_r,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    apple = _rest_on(peach, pear, APPLE)
    orange = _rest_on(apple, pear, ORANGE)

    assert _vertical_order_penalty([peach, pear, apple, orange]) == 0.0


# --- Stage gate -------------------------------------------------------


def test_inversion_fraction_reads_the_board_order() -> None:
    """An ordered board is 0, a reversed board is 1. The caller passes sign."""
    ordered = [_on_floor(PEAR, 70.0), _on_floor(DEKOPON, 200.0), _on_floor(GRAPE, 320.0)]
    assert inversion_fraction(ordered, 1) == 0.0
    assert inversion_fraction(ordered, -1) == 1.0


def test_a_fruit_trapped_between_bigger_ones_reads_as_broken() -> None:
    """A fruit stuck in a valley is read as a broken board even if inverted with only one side.

    The order orange (4), grape (2), apple (5) (sign=-1, so the right is big).
    Counting only the left and right of pairs, the grape is inverted only against the orange, so
    it is 1/3 = 0.333, below the 0.35 threshold, and falls into 'tidy'.
    A fruit squeezed between two big fruits is out of place with respect to both sides,
    so counting both sides gives 2/3 = 0.667.
    """
    orange = _on_floor(ORANGE, 69.664)
    grape = _on_floor(GRAPE, 150.0)
    apple = _on_floor(APPLE, 230.336)
    fruits = [orange, apple, grape]

    assert _is_nestled(grape, fruits)
    assert inversion_fraction(fruits, -1) > 0.35
    assert board_is_broken(fruits, -1)


def test_board_is_broken_only_past_the_threshold() -> None:
    """Recovery rules are not applied on a tidy board.

    Only the direction of the gate is pinned. The threshold itself is a number decided by A/B.
    """
    ordered = [_on_floor(PEAR, 70.0), _on_floor(DEKOPON, 200.0), _on_floor(GRAPE, 320.0)]
    assert not board_is_broken(ordered, 1)
    assert board_is_broken(ordered, -1)
