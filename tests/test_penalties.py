"""Unit tests for the size-order penalty and its valley exemption.

Move selection is in tests/test_policy.py. This pins down the meaning of the penalty rules themselves.
The valley check (`_is_nestled`) and the condition that actually exempts within it (`_size_order_exempt`)
are different things, so they are kept separate.
"""

from src.observe import Observation
from src.penalties import _is_nestled, _size_order_exempt, _size_order_penalty
from src.policy import choose_x
from src.sim.sim_physics import simulate_drop_held
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit

PEAR, DEKOPON, GRAPE = 6, 3, 2
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
