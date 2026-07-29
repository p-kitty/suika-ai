"""Unit tests for the policy. No screen is used."""

from src.observe import Observation
from src.policy import _after_drop, _ideal_x, _land_y, _radius, choose_x
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


def test_empty_board_drops_near_ideal_for_size() -> None:
    # On an empty board, near the size-order ideal (cherry leans right).
    x = choose_x(_obs(held_type=0))
    assert abs(x - _ideal_x(0)) < 40


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


def test_small_fruit_goes_right_of_large() -> None:
    # A big fruit on the left. A small held cannot merge, so toward the right (size order).
    big_r = _radius(6)
    big = Fruit(type=6, x=90, y=NORMALIZED_HEIGHT - big_r, radius=big_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(big,)))
    assert x > NORMALIZED_WIDTH / 2


def test_prefers_held_that_enables_next_merge() -> None:
    # held=grape is not on the board and cannot merge. One cherry on the right. next is cherry too.
    # Placing held near the right cherry lets next merge. Placing it left is far.
    cherry_r = _radius(0)
    grape_r = _radius(2)
    cherry = Fruit(type=0, x=310, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    # There is floor on the left too, but choose the right for next's merge.
    x = choose_x(_obs(held_type=2, fruits=(cherry,), next_type=0))
    assert x > NORMALIZED_WIDTH / 2
    assert abs(x - cherry.x) < cherry_r + grape_r * 2 + 50


def test_orange_stacks_on_left_apple() -> None:
    # Apple at the left edge: if the floor right next to it is open, line up to the right in size order (beside over on top).
    apple_r = _radius(5)
    orange_r = _radius(4)
    apple = Fruit(type=5, x=apple_r + 8, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    side = apple.x + apple_r + orange_r
    x = choose_x(_obs(held_type=4, fruits=(apple,)))
    assert abs(x - side) < orange_r
    assert x > apple.x


def test_orange_stacks_on_apple_even_with_next_cherry() -> None:
    # Even if next is the cherry on the right, the orange goes right next to the apple.
    apple_r = _radius(5)
    orange_r = _radius(4)
    cherry_r = _radius(0)
    apple = Fruit(type=5, x=apple_r + 8, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    cherry = Fruit(type=0, x=300, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    side = apple.x + apple_r + orange_r
    x = choose_x(_obs(held_type=4, fruits=(apple, cherry), next_type=0))
    assert abs(x - side) < orange_r * 1.5
    assert x > apple.x
    assert x < NORMALIZED_WIDTH / 2


def test_orange_on_top_when_ordered_side_blocked() -> None:
    # When the spot right of the apple is blocked, stack on top.
    apple_r = _radius(5)
    orange_r = _radius(4)
    grape_r = _radius(2)
    apple = Fruit(type=5, x=apple_r + 8, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    side_x = apple.x + apple_r + orange_r
    blocker = Fruit(type=2, x=side_x, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    x = choose_x(_obs(held_type=4, fruits=(apple, blocker)))
    assert abs(x - apple.x) < apple_r * 0.9
