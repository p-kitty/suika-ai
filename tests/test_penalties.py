"""Unit tests for the size-order penalty, its valley exemption and how merges are pushed.

Move selection is in tests/test_policy.py. This pins down the meaning of the penalty rules themselves.
The valley check (`_is_nestled`) and the condition that actually exempts within it (`_size_order_exempt`)
are different things, so they are kept separate.
"""

from src.observe import Observation
from src.penalties import (
    MERGE_BIG_SIDE_SLACK_FRAC,
    STRANDED_DROP_WEIGHT,
    _perch_penalty,
    _is_nestled,
    _size_order_exempt,
    _size_order_penalty,
    merge_lands_big_side,
    stranded_drop_penalty,
)
from src.policy import choose_x
from src.sim.sim_physics import simulate_drop_held
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit

PEAR, DEKOPON, GRAPE, STRAW, CHERRY = 6, 3, 2, 1, 0
PEACH = 7
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
    """A fruit in a valley is excluded from size order only when a partner remains in the same valley.

    In both boards the grape (2) is left of the dekopon (3) = inverted, and the valley shape is the same.
    The only difference is whether another grape, the merge partner, is inside the valley. Without a partner
    there is no prospect of leaving the valley, so it counts as a plain ordering violation.
    """
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)
    dekopon = _on_floor(DEKOPON, 230.0)
    alone = [pear, grape, dekopon]
    with_partner = [pear, grape, _on_floor(GRAPE, 200.0), dekopon]

    assert _is_nestled(grape, alone)
    assert _is_nestled(grape, with_partner)
    assert not _size_order_exempt(grape, alone)
    assert _size_order_exempt(grape, with_partner)


def test_valley_fruit_is_not_exempt_by_a_partner_outside_the_valley() -> None:
    """A partner outside the valley does not exempt. The big wall fruits keep them from meeting.

    Move 35 of seed=834761 had this shape (a strawberry left in the valley of a pear and a pineapple
    was exempted on the basis of a strawberry at the opposite edge).
    """
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)
    dekopon = _on_floor(DEKOPON, 230.0)
    outside = [pear, grape, dekopon, _on_floor(GRAPE, 330.0)]

    assert _is_nestled(grape, outside)
    assert not _size_order_exempt(grape, outside)


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
    after, _merges, _types, _held_merged, _held_fruit = simulate_drop_held(
        list(fruits), DEKOPON, choose_x(obs)
    )
    dekopon = next(f for f in after if f.type == DEKOPON)
    grape = next(f for f in after if f.type == GRAPE)

    assert dekopon.x < grape.x


def test_merge_lands_big_side_follows_the_board_direction() -> None:
    """The same movement flips pass/fail depending on which side is big."""
    held_r = fruit_radius(DEKOPON)
    moved_right = _on_floor(DEKOPON, 200.0 + held_r * (MERGE_BIG_SIDE_SLACK_FRAC + 0.1))

    assert merge_lands_big_side(200.0, moved_right, held_r, -1)
    assert not merge_lands_big_side(200.0, moved_right, held_r, LARGE_LEFT)


def test_merge_lands_big_side_ignores_a_shift_under_the_slack() -> None:
    """The merge position is the midpoint of the two centers, so a shift under a radius does not count as pushed."""
    held_r = fruit_radius(DEKOPON)
    slack = held_r * MERGE_BIG_SIDE_SLACK_FRAC

    assert not merge_lands_big_side(200.0, _on_floor(DEKOPON, 200.0 + slack - 1.0), held_r, -1)
    assert merge_lands_big_side(200.0, _on_floor(DEKOPON, 200.0 + slack + 1.0), held_r, -1)


def test_merge_lands_big_side_needs_a_surviving_fruit() -> None:
    """When it grows into a watermelon and disappears there is nowhere it was pushed to (held_fruit is None)."""
    assert not merge_lands_big_side(200.0, None, fruit_radius(DEKOPON), -1)


def test_stranded_drop_costs_more_the_bigger_the_walls() -> None:
    """Heavier the wider the type gap to the valley walls. How unrecoverable it is applies directly.

    What sets the weight is **the smaller of the left and right** walls (here the dekopon).
    Measuring by the bigger one would make even shapes recoverable because one side is low heavy.
    """
    # The valley width is judged by the radius of the dropped fruit (`_valley_flanks`), so
    # the spacing is set to one counted as a valley even for a cherry.
    pear = _on_floor(PEAR, 70.0)
    dekopon = _on_floor(DEKOPON, 200.0)
    straw = _on_floor(STRAW, 150.0)
    cherry = _on_floor(CHERRY, 150.0)

    shallow = stranded_drop_penalty([pear, straw, dekopon], straw)
    deep = stranded_drop_penalty([pear, cherry, dekopon], cherry)

    assert shallow == STRANDED_DROP_WEIGHT
    assert deep == STRANDED_DROP_WEIGHT * 2


def test_stranded_drop_is_free_with_a_partner_in_the_same_valley() -> None:
    """With a partner in the same valley it can merge, so it is not stranded. Does not crush valley growing."""
    pear = _on_floor(PEAR, 70.0)
    straw = _on_floor(STRAW, 170.0)
    partner = _on_floor(STRAW, 200.0)
    dekopon = _on_floor(DEKOPON, 230.0)

    assert stranded_drop_penalty([pear, straw, partner, dekopon], straw) == 0.0


def test_stranded_drop_ignores_a_partner_outside_the_valley() -> None:
    """A partner outside the valley is blocked by the big wall fruits, so it stays stranded."""
    pear = _on_floor(PEAR, 70.0)
    straw = _on_floor(STRAW, 170.0)
    dekopon = _on_floor(DEKOPON, 230.0)
    outside = _on_floor(STRAW, 330.0)

    assert stranded_drop_penalty([pear, straw, dekopon, outside], straw) > 0.0


def test_stranded_drop_needs_walls_bigger_than_the_threshold() -> None:
    """Merely being wedged one tier up is not stranded. The next merge fixes the order."""
    dekopon_left = _on_floor(DEKOPON, 90.0)
    grape = _on_floor(GRAPE, 160.0)
    dekopon_right = _on_floor(DEKOPON, 220.0)

    assert stranded_drop_penalty([dekopon_left, grape, dekopon_right], grape) == 0.0


def test_perch_is_free_in_a_one_step_notch() -> None:
    """A hollow in a wall one tier up is the next rung. Count a perch only when placed on a bare top.

    When the board fills with big fruits the type gap opens on every shoulder, so without the exemption a small fruit has no escape,
    and the policy tips toward avoiding shoulders and roofing another small fruit
    (move 72 of seed=890270).
    """
    peach = _on_floor(PEACH, 78.0)
    top_y = peach.y - peach.radius - fruit_radius(GRAPE)
    grape = Fruit(type=GRAPE, x=120.0, y=top_y, radius=fruit_radius(GRAPE), confidence=90)
    wall = Fruit(type=DEKOPON, x=180.0, y=top_y, radius=fruit_radius(DEKOPON), confidence=90)

    assert _perch_penalty([peach, grape]) > 0.0
    assert _perch_penalty([peach, grape, wall]) == 0.0


def test_perch_still_counts_a_deep_valley() -> None:
    """A valley with a wide type gap is a trap, not a rung. Keep counting it as a perch.

    The cherry in the position that motivated `_perch_penalty` was also sitting in the valley between apple and orange.
    Exempting valleys would erase that very case.
    """
    pine = _on_floor(8, 150.0)
    top_y = pine.y - pine.radius - fruit_radius(CHERRY)
    cherry = Fruit(type=CHERRY, x=150.0, y=top_y, radius=fruit_radius(CHERRY), confidence=90)
    walls = [
        Fruit(type=t, x=150.0 + dx, y=top_y, radius=fruit_radius(t), confidence=90)
        for t, dx in ((5, -60.0), (4, 60.0))
    ]

    assert _perch_penalty([pine, cherry, *walls]) > 0.0
