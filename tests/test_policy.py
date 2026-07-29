"""Unit tests for the policy. No screen is used."""

from src.observe import Observation
from src.policy import _after_drop, _land_y, _radius, choose_x
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit


def _obs(*, held_type: int, fruits: tuple[Fruit, ...] = (), next_type: int | None = None) -> Observation:
    return Observation(
        ready=True,
        blocked=False,
        fruits=fruits,
        held_type=held_type,
        held_x=NORMALIZED_WIDTH / 2,
        next_type=next_type,
    )


def test_empty_board_drops_near_center() -> None:
    x = choose_x(_obs(held_type=0))
    assert abs(x - NORMALIZED_WIDTH / 2) < 40


def test_prefers_same_type_over_empty_low_column() -> None:
    # A same type on the right, the left open but only floor. Choose on / near the same type.
    cherry_r = _radius(0)
    same = Fruit(type=0, x=280, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(same,)))
    # Either directly on top or beside it. The same-type side rather than the open floor on the left.
    assert abs(x - same.x) < cherry_r * 3
    assert x > 200


def test_avoids_dangerous_tall_stack() -> None:
    # The left is stacked nearly to the ceiling. The right is open without a low same type. Go low.
    big_r = _radius(5)
    tall = Fruit(type=5, x=80, y=60 + big_r, radius=big_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(tall,)))
    assert abs(x - tall.x) > 80
    assert x >= NORMALIZED_WIDTH / 2


def test_land_y_on_floor_when_empty() -> None:
    held_r = _radius(0)
    assert abs(_land_y((), 200, held_r) - (NORMALIZED_HEIGHT - held_r)) < 1e-6


def test_land_y_rests_on_fruit() -> None:
    held_r = _radius(0)
    fruit = Fruit(type=1, x=200, y=400, radius=20, confidence=90)
    land = _land_y((fruit,), 200, held_r)
    assert abs(land - (fruit.y - fruit.radius - held_r)) < 1e-6


def test_prefers_merge_that_lowers_stack() -> None:
    # Two same types slightly apart on the right; dropping between them merges. The left is open floor. Choose the merge column.
    cherry_r = _radius(0)
    floor_y = NORMALIZED_HEIGHT - cherry_r
    # A non-touching spacing (wider than 2r + CONTACT, a distance where a third reaches both).
    a = Fruit(type=0, x=250, y=floor_y, radius=cherry_r, confidence=90)
    b = Fruit(type=0, x=250 + 2 * cherry_r + 8, y=floor_y, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(a, b)))
    assert x > 200
    after, merges = _after_drop(_obs(held_type=0, fruits=(a, b)), x)
    assert merges >= 1
    assert abs(x - (a.x + b.x) / 2) < cherry_r * 4


def test_does_not_bury_same_type_under_different() -> None:
    # A cherry on the left. held is strawberry, and dropping left buries it.
    # There is a strawberry on the right, so merge on the right.
    cherry_r = _radius(0)
    straw_r = _radius(1)
    floor_cherry = NORMALIZED_HEIGHT - cherry_r
    floor_straw = NORMALIZED_HEIGHT - straw_r
    buried = Fruit(type=0, x=100, y=floor_cherry, radius=cherry_r, confidence=90)
    mate = Fruit(type=1, x=300, y=floor_straw, radius=straw_r, confidence=90)
    x = choose_x(_obs(held_type=1, fruits=(buried, mate)))
    assert abs(x - mate.x) < straw_r * 3
    assert x > 200


def test_sets_up_next_when_no_immediate_merge() -> None:
    # held is grape. No grape on the board. next is cherry, with a cherry on the right.
    # Place it toward the right to set up next's merge.
    cherry_r = _radius(0)
    grape_r = _radius(2)
    target = Fruit(type=0, x=300, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    # With only a big obstacle on the left, the low next setup on the right wins.
    wall = Fruit(type=5, x=80, y=NORMALIZED_HEIGHT - _radius(5), radius=_radius(5), confidence=90)
    x = choose_x(_obs(held_type=2, fruits=(target, wall), next_type=0))
    assert x > NORMALIZED_WIDTH / 2
    # Near the next cherry (place the grape beside it).
    assert abs(x - target.x) < cherry_r + grape_r * 2 + 40
