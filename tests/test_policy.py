"""Unit tests for the policy. No screen is used."""

from src.observe import Observation
from src.policy import (
    _after_drop,
    _chain_center_gap,
    _ideal_x,
    _land_y,
    _preview_land,
    _radius,
    _score,
    choose_x,
)
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


def test_prefers_merging_wedged_small_over_stacking_on_larger() -> None:
    # A grape wedged between two apples. held is grape too.
    # Even if the orange's lining-up side on the right is open, merging the wedged same type takes priority.
    apple_r = _radius(5)
    grape_r = _radius(2)
    orange_r = _radius(4)
    floor_apple = NORMALIZED_HEIGHT - apple_r
    floor_grape = NORMALIZED_HEIGHT - grape_r
    floor_orange = NORMALIZED_HEIGHT - orange_r
    left = Fruit(type=5, x=apple_r + 10, y=floor_apple, radius=apple_r, confidence=90)
    mid_x = left.x + apple_r + grape_r - 2
    wedged = Fruit(type=2, x=mid_x, y=floor_grape, radius=grape_r, confidence=90)
    right = Fruit(type=5, x=mid_x + grape_r + apple_r - 2, y=floor_apple, radius=apple_r, confidence=90)
    orange = Fruit(
        type=4,
        x=min(NORMALIZED_WIDTH - orange_r - 5, right.x + apple_r + orange_r + 30),
        y=floor_orange,
        radius=orange_r,
        confidence=90,
    )
    x = choose_x(_obs(held_type=2, fruits=(left, wedged, right, orange)))
    assert abs(x - wedged.x) < grape_r * 2
    after, merges = _after_drop(_obs(held_type=2, fruits=(left, wedged, right, orange)), x)
    assert merges >= 1


def test_stacks_strawberry_on_grape_when_next_is_strawberry() -> None:
    # NOTES: when wanting to grow a grape and held/next are strawberries, on top of the grape.
    # Even if the neighboring floor is open, choose on top of the target grown with a same-type next.
    grape_r = _radius(2)
    straw_r = _radius(1)
    grape = Fruit(type=2, x=120, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    x = choose_x(_obs(held_type=1, fruits=(grape,), next_type=1))
    assert abs(x - grape.x) < grape_r * 0.85


def test_grows_grape_even_with_distant_strawberry_merge() -> None:
    # Even with a strawberry on the right that could merge immediately, growing the grape on the left takes priority.
    grape_r = _radius(2)
    straw_r = _radius(1)
    grape = Fruit(type=2, x=100, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    lone = Fruit(type=1, x=320, y=NORMALIZED_HEIGHT - straw_r, radius=straw_r, confidence=90)
    x = choose_x(_obs(held_type=1, fruits=(grape, lone), next_type=1))
    assert abs(x - grape.x) < grape_r * 0.85
    assert abs(x - lone.x) > straw_r * 3


def test_stacks_grape_on_dekopon_when_next_is_grape() -> None:
    # The same pattern one tier up: if held/next are grapes, on top of the dekopon.
    dek_r = _radius(3)
    grape_r = _radius(2)
    dek = Fruit(type=3, x=110, y=NORMALIZED_HEIGHT - dek_r, radius=dek_r, confidence=90)
    x = choose_x(_obs(held_type=2, fruits=(dek,), next_type=2))
    assert abs(x - dek.x) < dek_r * 0.85


def test_orange_leaves_room_for_dekopon_beside_strawberry() -> None:
    # Cherry and strawberry on the right. Placing the orange right left of the strawberry leaves
    # no column for dekopon and grape to line up, so keep the intermediate stages' distance.
    cherry_r = _radius(0)
    straw_r = _radius(1)
    orange_r = _radius(4)
    cherry = Fruit(
        type=0,
        x=NORMALIZED_WIDTH - cherry_r - 2,
        y=NORMALIZED_HEIGHT - cherry_r,
        radius=cherry_r,
        confidence=90,
    )
    # A strawberry left of the ideal orange: the old policy tended to place right beside it.
    straw = Fruit(type=1, x=280, y=NORMALIZED_HEIGHT - straw_r, radius=straw_r, confidence=90)
    x = choose_x(_obs(held_type=4, fruits=(cherry, straw)))
    need = _chain_center_gap(4, 1)
    assert straw.x - x >= need - 4
    # Clearly apart rather than right at contact.
    assert straw.x - x - straw_r - orange_r > _radius(2) + _radius(3)


def test_grows_toward_large_fruit_on_the_right() -> None:
    # Even after big fruits gathered on the right, do not re-grow the left big. The apple goes right left of the pear on the right.
    pear_r = _radius(6)
    apple_r = _radius(5)
    pear = Fruit(type=6, x=300, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    side = pear.x - pear_r - apple_r
    x = choose_x(_obs(held_type=5, fruits=(pear,)))
    assert abs(x - side) < apple_r
    assert x < pear.x


def test_drop_on_slope_rolls_to_floor() -> None:
    # Dropping on a big fruit's right shoulder rolls down its side and lands on the floor.
    pear_r = _radius(6)
    cherry_r = _radius(0)
    pear = Fruit(type=6, x=200, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    drop_x = pear.x + pear_r * 0.55
    land_x, land_y = _preview_land((pear,), 0, drop_x, cherry_r)
    assert land_x > drop_x
    assert land_y >= NORMALIZED_HEIGHT - cherry_r - 1.0
    assert land_x >= pear.x + pear_r + cherry_r - 2.0


def test_merge_result_settles_from_midpoint() -> None:
    # A merged fruit appears at the midpoint and then lands (leaning toward the sticking direction + rolling).
    cherry_r = _radius(0)
    floor_y = NORMALIZED_HEIGHT - cherry_r
    a = Fruit(type=0, x=200, y=floor_y, radius=cherry_r, confidence=90)
    b = Fruit(type=0, x=200 + 2 * cherry_r + 4, y=floor_y, radius=cherry_r, confidence=90)
    mid = (a.x + b.x) / 2
    after, merges = _after_drop(_obs(held_type=0, fruits=(a, b)), mid)
    assert merges >= 1
    grown = [f for f in after if f.type == 1]
    assert grown
    assert abs(grown[0].x - mid) < cherry_r * 2


def test_strawberry_does_not_roll_left_of_grape() -> None:
    # Early on: placing at the upper left of a grape rolls left and breaks the size order. Choose right beside it or on top.
    grape_r = _radius(2)
    straw_r = _radius(1)
    grape = Fruit(type=2, x=160, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    obs = _obs(held_type=1, fruits=(grape,))
    x = choose_x(obs)
    land_x, _land = _preview_land((grape,), 1, x, straw_r)
    assert land_x >= grape.x - 1.0
    # A left-shoulder drop rolls onto the floor left of the grape, so it is not chosen.
    left_shoulder = grape.x - grape_r * 0.5
    assert _score(obs, left_shoulder, straw_r) < _score(obs, x, straw_r)


def test_left_shoulder_of_grape_rolls_to_left_floor() -> None:
    # The simulation itself reproduces 'left shoulder → rolls left'.
    grape_r = _radius(2)
    straw_r = _radius(1)
    grape = Fruit(type=2, x=160, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    drop_x = grape.x - grape_r * 0.5
    land_x, land_y = _preview_land((grape,), 1, drop_x, straw_r)
    assert land_x < grape.x - grape_r
    assert land_y >= NORMALIZED_HEIGHT - straw_r - 1.0


def test_grape_stays_beside_right_edge_strawberry() -> None:
    # Move 1 strawberry at the right edge, move 2 grape lands stably right beside it on the left.
    # Do not choose moves knocked to the left edge by contact (the line where a dekopon cannot be placed afterward and it collapses).
    straw_r = _radius(1)
    grape_r = _radius(2)
    straw = Fruit(
        type=1,
        x=NORMALIZED_WIDTH - straw_r - 2,
        y=NORMALIZED_HEIGHT - straw_r,
        radius=straw_r,
        confidence=90,
    )
    obs = _obs(held_type=2, fruits=(straw,))
    x = choose_x(obs)
    land_x, land_y = _preview_land((straw,), 2, x, grape_r)
    assert land_y >= NORMALIZED_HEIGHT - grape_r - 1.0
    assert land_x > NORMALIZED_WIDTH * 0.5
    assert abs(land_x - (straw.x - straw_r - grape_r - 4.0)) < grape_r
    # A neighbor with a gap beats a drop that hits the shoulder and slides to the left edge.
    shoulder = straw.x - straw_r * 0.4
    assert _score(obs, x, grape_r) > _score(obs, shoulder, grape_r)


def test_pushes_near_orange_pair_from_outside() -> None:
    # Two close oranges: push from the left outside to join them rather than stacking on top.
    # With a big fruit on the right, pushing from the right outside is unfavorable (same type as doko2).
    orange_r = _radius(4)
    apple_r = _radius(5)
    pear_r = _radius(6)
    floor_o = NORMALIZED_HEIGHT - orange_r
    left = Fruit(type=4, x=70, y=floor_o, radius=orange_r, confidence=90)
    right = Fruit(type=4, x=70 + 2 * orange_r + 20, y=floor_o, radius=orange_r, confidence=90)
    apple = Fruit(
        type=5,
        x=right.x + orange_r + apple_r + 10,
        y=NORMALIZED_HEIGHT - apple_r,
        radius=apple_r,
        confidence=90,
    )
    pear = Fruit(
        type=6,
        x=apple.x + apple_r + pear_r + 10,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    obs = _obs(held_type=3, fruits=(left, right, apple, pear), next_type=4)
    x = choose_x(obs)
    assert x < left.x
    assert x <= left.x - left.radius * 0.45
