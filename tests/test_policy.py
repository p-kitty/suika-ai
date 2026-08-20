"""Unit tests for the thin bootstrap policy. No screen is used.

Concrete procedures (push-ins, cascade gap opening) are not required.
Only move selection for merging, danger avoidance, burying, rolling accidents and valley growing is pinned.
The fall physics itself is in tests/test_sim_physics.py.
"""

import math

from src.observe import Observation, clamp_drop_x
from src.penalties import (
    FOREIGN_AIM_CENTER_FRAC,
    FOREIGN_AIM_PENALTY,
    center_tiebreak,
    foreign_aim_penalty,
    ideal_x,
)
from src.policy import _candidates, _score, choose_x
from src.reward import is_lost, merge_points
from src.sim.sim_physics import landed_xy, preview_land, simulate_drop, simulate_drop_held
from src.vision.classify import fruit_radius
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
    x = choose_x(_obs(held_type=0))
    assert abs(x - ideal_x(0)) < 40


def test_prefers_same_type_over_empty_low_column() -> None:
    cherry_r = fruit_radius(0)
    same = Fruit(type=0, x=280, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(same,)))
    assert abs(x - same.x) < cherry_r * 3
    assert x > 200


def test_avoids_dangerous_tall_stack() -> None:
    """Settle away from a dangerous pile. The aimed column (x) itself does not matter (same reason as above)."""
    big_r = fruit_radius(5)
    cherry_r = fruit_radius(0)
    tall = Fruit(type=5, x=80, y=60 + big_r, radius=big_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(tall,)))
    after, _merges, _types = simulate_drop((tall,), 0, x)
    cherry = next(f for f in after if f.type == 0)
    assert abs(cherry.x - tall.x) > big_r + cherry_r * 3


def test_prefers_merge_that_lowers_stack() -> None:
    cherry_r = fruit_radius(0)
    floor_y = NORMALIZED_HEIGHT - cherry_r
    a = Fruit(type=0, x=250, y=floor_y, radius=cherry_r, confidence=90)
    b = Fruit(type=0, x=250 + 2 * cherry_r + 8, y=floor_y, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(a, b)))
    assert x > 200
    _after, merges, _types = simulate_drop((a, b), 0, x)
    assert merges >= 1
    assert abs(x - (a.x + b.x) / 2) < cherry_r * 4


def test_does_not_bury_same_type_under_different() -> None:
    cherry_r = fruit_radius(0)
    straw_r = fruit_radius(1)
    floor_cherry = NORMALIZED_HEIGHT - cherry_r
    floor_straw = NORMALIZED_HEIGHT - straw_r
    buried = Fruit(type=0, x=100, y=floor_cherry, radius=cherry_r, confidence=90)
    mate = Fruit(type=1, x=300, y=floor_straw, radius=straw_r, confidence=90)
    obs = _obs(held_type=1, fruits=(buried, mate))
    # A move toward a mate beats stacking directly above a different type.
    assert _score(obs, mate.x, straw_r) > _score(obs, buried.x, straw_r)
    x = choose_x(obs)
    after, merges, _types = simulate_drop((buried, mate), 1, x)
    # Do not pick burying directly above. Grazing, rolling and merging with the mate is fine.
    stacked = [
        f
        for f in after
        if f.type == 1 and abs(f.x - buried.x) < straw_r * 0.5
    ]
    assert not stacked
    assert merges >= 1 or abs(x - buried.x) > straw_r * 0.5


def test_sets_up_next_when_no_immediate_merge() -> None:
    cherry_r = fruit_radius(0)
    grape_r = fruit_radius(2)
    target = Fruit(type=0, x=300, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    wall = Fruit(type=5, x=80, y=NORMALIZED_HEIGHT - fruit_radius(5), radius=fruit_radius(5), confidence=90)
    fruits = (target, wall)
    x = choose_x(_obs(held_type=2, fruits=fruits, next_type=0))
    # It gets knocked by collisions, so judge by the actual landing position, not the aimed column x.
    after, _merges, _merge_types, held_merged = simulate_drop_held(fruits, 2, x)
    land_x, _land_y = landed_xy(fruits, after, 2, x, grape_r, held_merged)
    assert abs(land_x - target.x) < cherry_r + grape_r * 2 + 40


def test_small_fruit_goes_right_of_large() -> None:
    """Settle away from a big fruit. The aimed column (x) itself does not matter.

    Aiming right next to a big fruit is as intended if it rolls and settles away from it.
    Comparing the aim point with an absolute value of NORMALIZED_WIDTH keeps failing even when an unrelated
    column is chosen, because of the rolling (measured: aiming at x=132 rolls to x=384).
    """
    big_r = fruit_radius(6)
    cherry_r = fruit_radius(0)
    big = Fruit(type=6, x=90, y=NORMALIZED_HEIGHT - big_r, radius=big_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(big,)))
    after, _merges, _types = simulate_drop((big,), 0, x)
    cherry = next(f for f in after if f.type == 0)
    assert abs(cherry.x - big.x) > big_r + cherry_r * 3


def test_prefers_held_that_enables_next_merge() -> None:
    cherry_r = fruit_radius(0)
    grape_r = fruit_radius(2)
    cherry = Fruit(type=0, x=310, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=2, fruits=(cherry,), next_type=0))
    assert x > NORMALIZED_WIDTH / 2
    assert abs(x - cherry.x) < cherry_r + grape_r * 2 + 50


def test_grows_valley_fruit_when_held_and_next_are_one_smaller() -> None:
    # Grape in the valley, held/next are strawberries (one below the valley fruit). Dropping both makes a grape
    # that merges with the grape in the valley, so place it in the valley instead of escaping to the corner.
    from src.penalties import valley_grow_ok

    orange_r = fruit_radius(4)
    apple_r = fruit_radius(5)
    grape_r = fruit_radius(2)
    straw_r = fruit_radius(1)
    sep = orange_r + apple_r + grape_r * 2 + 20.0
    center = 150.0
    left = Fruit(
        type=4,
        x=center - sep / 2,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    right = Fruit(
        type=5,
        x=center + sep / 2,
        y=NORMALIZED_HEIGHT - apple_r,
        radius=apple_r,
        confidence=90,
    )
    grape = Fruit(
        type=2, x=center, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90
    )
    fruits = (left, right, grape)
    obs = _obs(held_type=1, fruits=fruits, next_type=1)
    land_x, _land_y = preview_land(fruits, 1, choose_x(obs), straw_r)
    assert left.x < land_x < right.x
    assert valley_grow_ok(fruits, land_x, 1, 1)


def test_does_not_grow_smaller_junk_in_valley() -> None:
    # Do not add a fruit smaller than the junk in the valley, making it impossible to clean up.
    pear_r = fruit_radius(6)
    grape_r = fruit_radius(2)
    straw_r = fruit_radius(1)
    left = Fruit(type=6, x=150, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    right = Fruit(
        type=6,
        x=150 + pear_r * 2 + grape_r * 1.5,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    cx = (left.x + right.x) / 2
    dx = cx - left.x
    gy = left.y - math.sqrt((pear_r + grape_r) ** 2 - dx * dx)
    grape = Fruit(type=2, x=cx, y=gy, radius=grape_r, confidence=90)
    fruits = (left, right, grape)
    # If next is the same type, it waits to merge in the valley and legitimately scores high, so look with a different type.
    obs = _obs(held_type=1, fruits=fruits, next_type=3)
    valley = cx
    far = NORMALIZED_WIDTH - straw_r - 8
    assert _score(obs, far, straw_r) > _score(obs, valley, straw_r)


def test_strawberry_does_not_roll_left_of_grape() -> None:
    grape_r = fruit_radius(2)
    straw_r = fruit_radius(1)
    grape = Fruit(type=2, x=160, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    obs = _obs(held_type=1, fruits=(grape,))
    x = choose_x(obs)
    land_x, _land = preview_land((grape,), 1, x, straw_r)
    assert land_x >= grape.x - 1.0
    left_shoulder = grape.x - grape_r * 0.5
    assert _score(obs, left_shoulder, straw_r) < _score(obs, x, straw_r)


def test_grape_stays_beside_right_edge_strawberry() -> None:
    straw_r = fruit_radius(1)
    grape_r = fruit_radius(2)
    straw = Fruit(
        type=1,
        x=NORMALIZED_WIDTH - straw_r - 2,
        y=NORMALIZED_HEIGHT - straw_r,
        radius=straw_r,
        confidence=90,
    )
    obs = _obs(held_type=2, fruits=(straw,))
    x = choose_x(obs)
    land_x, land_y = preview_land((straw,), 2, x, grape_r)
    # Contact penetration can sink it a few px below the floor. Only check that it is not stacked directly above (~441).
    assert land_y >= NORMALIZED_HEIGHT - grape_r - 4.0
    assert land_x > NORMALIZED_WIDTH * 0.5
    assert abs(land_x - (straw.x - straw_r - grape_r)) < grape_r * 2
    shoulder = straw.x - straw_r * 0.4
    assert _score(obs, x, grape_r) > _score(obs, shoulder, grape_r)


def test_strawberry_stays_beside_right_edge_cherry() -> None:
    cherry_r = fruit_radius(0)
    straw_r = fruit_radius(1)
    cherry = Fruit(
        type=0,
        x=NORMALIZED_WIDTH - cherry_r,
        y=NORMALIZED_HEIGHT - cherry_r,
        radius=cherry_r,
        confidence=90,
    )
    obs = _obs(held_type=1, fruits=(cherry,))
    x = choose_x(obs)
    land_x, land_y = preview_land((cherry,), 1, x, straw_r)
    # With the floor Segment radius of 2px, the center stops above NORMALIZED_HEIGHT - r.
    assert land_y >= NORMALIZED_HEIGHT - straw_r - 3.0
    assert land_x > NORMALIZED_WIDTH * 0.5
    assert land_x < cherry.x
    above = NORMALIZED_WIDTH - straw_r
    above_land, _ = preview_land((cherry,), 1, above, straw_r)
    assert above_land < NORMALIZED_WIDTH * 0.25
    assert _score(obs, x, straw_r) > _score(obs, above, straw_r)


def test_prefers_open_same_type_when_column_blocked() -> None:
    # Prefer a same type on the open side over a same type blocked from above.
    peach_r = fruit_radius(7)
    cover_r = fruit_radius(5)
    left_base = Fruit(
        type=4,
        x=80,
        y=NORMALIZED_HEIGHT - fruit_radius(4),
        radius=fruit_radius(4),
        confidence=90,
    )
    right_base = Fruit(
        type=4,
        x=320,
        y=NORMALIZED_HEIGHT - fruit_radius(4),
        radius=fruit_radius(4),
        confidence=90,
    )
    left = Fruit(
        type=7,
        x=80,
        y=left_base.y - left_base.radius - peach_r,
        radius=peach_r,
        confidence=90,
    )
    right = Fruit(
        type=7,
        x=320,
        y=right_base.y - right_base.radius - peach_r,
        radius=peach_r,
        confidence=90,
    )
    cover = Fruit(
        type=5,
        x=left.x,
        y=left.y - peach_r - cover_r,
        radius=cover_r,
        confidence=90,
    )
    fruits = (left_base, right_base, left, right, cover)
    x = choose_x(_obs(held_type=7, fruits=fruits))
    assert simulate_drop(fruits, 7, x)[1] >= 1


def test_chooses_merge_for_sandwiched_same_type() -> None:
    pear_r = fruit_radius(6)
    grape_r = fruit_radius(2)
    left = Fruit(type=6, x=160, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    right = Fruit(
        type=6,
        x=160 + pear_r * 2 + 40,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    gy = NORMALIZED_HEIGHT - grape_r
    dy = gy - left.y
    gx = left.x + math.sqrt((pear_r + grape_r) ** 2 - dy * dy)
    grape = Fruit(type=2, x=gx, y=gy, radius=grape_r, confidence=90)
    fruits = (left, right, grape)
    x = choose_x(_obs(held_type=2, fruits=fruits))
    assert simulate_drop(fruits, 2, x)[1] >= 1


def test_does_not_block_waiting_pair_with_bigger_fruit() -> None:
    # Two grapes waiting to merge. Do not block that valley by wedging a big orange in.
    grape_r = fruit_radius(2)
    orange_r = fruit_radius(4)
    floor_y = NORMALIZED_HEIGHT - grape_r
    a = Fruit(type=2, x=150, y=floor_y, radius=grape_r, confidence=90)
    b = Fruit(type=2, x=150 + grape_r * 2 + 30, y=floor_y, radius=grape_r, confidence=90)
    valley = (a.x + b.x) / 2
    obs = _obs(held_type=4, fruits=(a, b))
    x = choose_x(obs)
    # The chosen move beats the move that drops into the valley.
    assert _score(obs, x, orange_r) > _score(obs, valley, orange_r)
    land_x, _land_y = preview_land((a, b), 4, x, orange_r)
    assert not (a.x < land_x < b.x)


def test_avoids_foreign_center_stack() -> None:
    # Prefer open floor or the neighbor over directly above a different type.
    apple_r = fruit_radius(5)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    obs = _obs(held_type=4, fruits=(apple,))
    x = choose_x(obs)
    assert abs(x - apple.x) > apple_r * FOREIGN_AIM_CENTER_FRAC
    assert _score(obs, x, orange_r) > _score(obs, apple.x, orange_r)


def test_foreign_aim_ignores_buried_foreign() -> None:
    # Even in the center column of a different type below, no penalty if the geometric contact directly below is the shoulder of the fruit above.
    apple_r = fruit_radius(5)
    grape_r = fruit_radius(2)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    grape_x = apple.x + apple_r * 0.45
    grape_y = apple.y - math.sqrt((apple_r + grape_r) ** 2 - (grape_x - apple.x) ** 2)
    grape = Fruit(type=2, x=grape_x, y=grape_y, radius=grape_r, confidence=90)
    # Aim at the apple's center column, but what it touches directly below is the offset grape.
    assert foreign_aim_penalty((apple, grape), apple.x, 4, orange_r) == 0.0
    # Aiming straight down onto the head of a different type is penalized.
    assert foreign_aim_penalty((apple,), apple.x, 4, orange_r) == FOREIGN_AIM_PENALTY


def test_foreign_aim_ok_when_same_type_below() -> None:
    # If directly below is the same type it is waiting to merge, so no FOREIGN_AIM (not a merges condition).
    orange_r = fruit_radius(4)
    floor = NORMALIZED_HEIGHT - orange_r
    mate = Fruit(type=4, x=200, y=floor, radius=orange_r, confidence=90)
    assert foreign_aim_penalty((mate,), mate.x, 4, orange_r) == 0.0


def test_foreign_aim_penalizes_foreign_below_even_if_near_same_type() -> None:
    # Directly below is a different type. Even with a same type beside it, aiming directly above is penalized.
    apple_r = fruit_radius(5)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    mate = Fruit(
        type=4,
        x=200 + apple_r + orange_r - 4,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    assert foreign_aim_penalty((apple, mate), apple.x, 4, orange_r) == FOREIGN_AIM_PENALTY


def test_merges_when_three_same_type_waiting() -> None:
    # On a board with 3 of the same type, merge early with the same type in hand.
    cherry_r = fruit_radius(0)
    floor = NORMALIZED_HEIGHT - cherry_r
    fruits = (
        Fruit(type=0, x=120, y=floor, radius=cherry_r, confidence=90),
        Fruit(type=0, x=200, y=floor, radius=cherry_r, confidence=90),
        Fruit(type=0, x=280, y=floor, radius=cherry_r, confidence=90),
    )
    obs = _obs(held_type=0, fruits=fruits)
    x = choose_x(obs)
    after, merges, _types = simulate_drop(fruits, 0, x)
    assert merges >= 1
    # 3 + 1 advances merging; cherries decrease and higher fruits remain.
    assert sum(1 for f in after if f.type == 0) <= 2
    assert any(f.type >= 1 for f in after)
    # Merging is clearly better than dumping it at the edge as a fourth.
    far = 40.0
    assert _score(obs, x, cherry_r) > _score(obs, far, cherry_r) + 20.0


def test_biggest_prefers_edge_over_center() -> None:
    # Pushing the biggest fruit to the big side is left to size-order / ideal. A peach on an empty board leans left.
    peach_r = fruit_radius(7)
    obs = _obs(held_type=7)
    x = choose_x(obs)
    assert x < NORMALIZED_WIDTH * 0.45
    center = NORMALIZED_WIDTH / 2
    assert _score(obs, x, peach_r) > _score(obs, center, peach_r)


def test_large_fruits_prefer_clustering() -> None:
    # Big fruits stay close. For a peach at the left edge, the pear picks the neighbor over far away.
    peach_r = fruit_radius(7)
    pear_r = fruit_radius(6)
    peach = Fruit(
        type=7,
        x=peach_r + 4,
        y=NORMALIZED_HEIGHT - peach_r,
        radius=peach_r,
        confidence=90,
    )
    obs = _obs(held_type=6, fruits=(peach,))
    x = choose_x(obs)
    land_x, _land_y = preview_land((peach,), 6, x, pear_r)
    assert abs(land_x - peach.x) < peach_r + pear_r + 40
    far = NORMALIZED_WIDTH - pear_r - 8
    assert _score(obs, x, pear_r) > _score(obs, far, pear_r)


def _floor(fruit_type: int, x: float) -> Fruit:
    r = fruit_radius(fruit_type)
    return Fruit(type=fruit_type, x=x, y=NORMALIZED_HEIGHT - r, radius=r, confidence=90)


def _rest_on(a: Fruit, b: Fruit, fruit_type: int) -> Fruit:
    """A fruit of type resting on top touching both a and b. Scaffolding for tests building ladder shoulders."""
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


def _ladder_board() -> tuple[Fruit, ...]:
    """Corner peach + pear on the inside, apple and orange on the shoulders of those two (a completed ladder)."""
    peach_r, pear_r = fruit_radius(7), fruit_radius(6)
    peach = Fruit(
        type=7, x=peach_r + 2, y=NORMALIZED_HEIGHT - peach_r, radius=peach_r, confidence=90
    )
    pear = Fruit(
        type=6,
        x=peach.x + peach_r + pear_r,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    apple = _rest_on(peach, pear, 5)
    orange = _rest_on(apple, pear, 4)
    return (peach, pear, apple, orange)


def test_ladder_detects_full_stack() -> None:
    # peach→pear→apple→orange are all picked up as rungs.
    from src.ladder import find_anchor, rungs
    from src.policy import _order_sign

    fruits = _ladder_board()
    sign = _order_sign(fruits)
    anchor = find_anchor(fruits, sign)
    assert anchor is not None and anchor.type == 7
    assert sorted(rungs(fruits, anchor, sign)) == [4, 5, 6, 7]


def test_ladder_ignores_vertical_tower() -> None:
    # A pear stacked directly on top of the peach is not a ladder (a shape that collapses, so not counted as a rung).
    from src.ladder import find_anchor, rungs
    from src.policy import _order_sign

    peach_r, pear_r = fruit_radius(7), fruit_radius(6)
    peach = Fruit(
        type=7, x=peach_r + 2, y=NORMALIZED_HEIGHT - peach_r, radius=peach_r, confidence=90
    )
    tower = Fruit(
        type=6, x=peach.x, y=peach.y - peach_r - pear_r, radius=pear_r, confidence=90
    )
    fruits = (peach, tower)
    sign = _order_sign(fruits)
    anchor = find_anchor(fruits, sign)
    assert anchor is not None
    assert sorted(rungs(fruits, anchor, sign)) == [7]


def test_ladder_needs_no_ignition_hint() -> None:
    # Once built, the current choose_x ties with the best of an exhaustive sweep over x.
    # Pins down that guiding only the firing is pointless (if anything, add to the building side).
    from src.reward import merge_score

    fruits = _ladder_board()
    # Fill the floor to the right edge. If sparse, the pear gets pushed out and the shape does not hold.
    packed = list(fruits)
    cursor = fruits[1].x + fruits[1].radius
    for fruit_type in (6, 5, 5, 4, 4):
        r = fruit_radius(fruit_type)
        packed.append(
            Fruit(type=fruit_type, x=cursor + r, y=NORMALIZED_HEIGHT - r, radius=r, confidence=90)
        )
        cursor += 2 * r
    board = tuple(packed)

    best = max(
        merge_score(simulate_drop(board, 4, i * 8.0)[2])
        for i in range(NORMALIZED_WIDTH // 8 + 1)
    )
    chosen = merge_score(simulate_drop(board, 4, choose_x(_obs(held_type=4, fruits=board)))[2])
    assert best >= 100.0
    assert chosen >= best


def test_avoids_under_max_center_on_outer_edge() -> None:
    # Placing a small fruit at the big-side edge beyond the biggest is fine, but avoid the corner pocket below L's center.
    from src.penalties import _big_layout_penalty

    peach_r = fruit_radius(7)
    orange_r = fruit_radius(4)
    peach = Fruit(
        type=7,
        x=peach_r + 8,
        y=NORMALIZED_HEIGHT - peach_r,
        radius=peach_r,
        confidence=90,
    )
    # Left of the peach, on the floor (y > peach.y) = the big-side corner pocket (sign=+1).
    pocket = Fruit(
        type=4,
        x=orange_r + 2,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    assert pocket.x < peach.x
    assert pocket.y > peach.y
    # The peach's shoulder (edge side but y <= peach.y).
    shoulder_x = peach.x - peach_r * 0.4
    dx = abs(shoulder_x - peach.x)
    shoulder_y = peach.y - math.sqrt((peach_r + orange_r) ** 2 - dx * dx)
    shoulder = Fruit(
        type=4,
        x=shoulder_x,
        y=shoulder_y,
        radius=orange_r,
        confidence=90,
    )
    assert shoulder.y <= peach.y
    assert _big_layout_penalty((peach, pocket), sign=1) > _big_layout_penalty(
        (peach, shoulder), sign=1
    ) + 30
    # A small fruit on the small-side (right) floor does not make a corner pocket in the big-side layout.
    right_floor = Fruit(
        type=4,
        x=NORMALIZED_WIDTH - orange_r - 2,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    assert _big_layout_penalty((peach, right_floor), sign=1) < 20

    obs = _obs(held_type=4, fruits=(peach,))
    x = choose_x(obs)
    land_x, land_y = preview_land((peach,), 4, x, orange_r)
    # Do not drop into the floor pocket of the big-side corner.
    assert not (land_x < peach.x and land_y > peach.y)


def test_leaves_room_for_missing_rung_between_neighbours() -> None:
    """Pairs with a missing type in between are placed leaving that much gap.

    Pulling a grape right beside an opening orange leaves no place when a dekopon comes next,
    and it goes outside the grape, giving the order 4-2-3 (measured).
    """
    orange_r = fruit_radius(4)
    dekopon_r = fruit_radius(3)
    grape_r = fruit_radius(2)

    # Move 1 orange, move 2 grape (next is dekopon).
    x1 = choose_x(_obs(held_type=4, fruits=(), next_type=2))
    board1, _m, _t = simulate_drop((), 4, x1)
    x2 = choose_x(_obs(held_type=2, fruits=tuple(board1), next_type=3))
    board2, _m, _t = simulate_drop(board1, 2, x2)

    orange = next(f for f in board2 if f.type == 4)
    grape = next(f for f in board2 if f.type == 2)
    gap = abs(orange.x - grape.x) - orange_r - grape_r
    # Leave a gap close to one dekopon (80%, allowing for push-in margin).
    assert gap > dekopon_r * 2 * 0.8

    # Placing the dekopon on move 3 does not break size order.
    x3 = choose_x(_obs(held_type=3, fruits=tuple(board2)))
    board3, _m, _t = simulate_drop(board2, 3, x3)
    assert [f.type for f in sorted(board3, key=lambda f: f.x)] == [2, 3, 4]


def test_does_not_perch_small_fruit_on_biggest() -> None:
    """Do not put a fruit with a large type gap on top of a big-fruit pile.

    The board is move 97 of seed=642746 (NOTES 'Resolved: putting small fruits on a big fruit's shoulder').
    On a board where every placement buries something, only putting it on the pineapple pile had zero penalty.
    The aimed column does not matter; it only checks that the cherry does not settle inside the pineapple's footprint
    above the pineapple's center.
    """
    fruits = (
        Fruit(type=5, x=51.2, y=298.8, radius=fruit_radius(5), confidence=90),
        Fruit(type=8, x=78.4, y=421.6, radius=fruit_radius(8), confidence=90),
        Fruit(type=4, x=144.6, y=327.7, radius=fruit_radius(4), confidence=90),
        Fruit(type=2, x=208.7, y=337.8, radius=fruit_radius(2), confidence=90),
        Fruit(type=7, x=221.9, y=430.7, radius=fruit_radius(7), confidence=90),
        Fruit(type=6, x=290.2, y=328.3, radius=fruit_radius(6), confidence=90),
        Fruit(type=4, x=324.4, y=459.6, radius=fruit_radius(4), confidence=90),
        Fruit(type=2, x=371.6, y=317.4, radius=fruit_radius(2), confidence=90),
        Fruit(type=0, x=383.9, y=278.8, radius=fruit_radius(0), confidence=90),
    )
    pine = fruits[1]
    cherry_r = fruit_radius(0)
    x = choose_x(_obs(held_type=0, fruits=fruits, next_type=2))
    after, _merges, _types, held_merged = simulate_drop_held(fruits, 0, x)
    land_x, land_y = landed_xy(fruits, after, 0, x, cherry_r, held_merged)
    on_pine = abs(land_x - pine.x) <= pine.radius + cherry_r
    assert not (on_pine and land_y + cherry_r <= pine.y)


def test_does_not_kill_itself_when_a_surviving_drop_exists() -> None:
    """While a living move exists, do not choose a dying one.

    The board is move 172 of seed=982108 (NOTES 'Replaced the dangerous height slope with a filter').
    A position with 2 lethal and 31 surviving candidates, where a lethal move wins on eval by 23.7.
    Back when death was expressed as a penalty, it stacked an orange on the left-edge pile here and destroyed itself.
    The aimed column does not matter; it only checks that the board after the drop does not cross the losing line.
    """
    raw = (
        (3, 368.3, 126.3), (5, 51.2, 130.5), (7, 330.6, 215.9), (8, 78.4, 253.2),
        (1, 205.4, 295.7), (1, 298.5, 297.9), (3, 251.5, 316.2), (0, 383.9, 328.0),
        (7, 323.0, 382.4), (2, 28.3, 395.9), (9, 161.8, 403.0), (4, 40.8, 459.6),
        (3, 368.3, 468.3),
    )
    fruits = tuple(
        Fruit(type=t, x=x, y=y, radius=fruit_radius(t), confidence=90) for t, x, y in raw
    )
    x = choose_x(_obs(held_type=4, fruits=fruits, next_type=0))
    after, _merges, _types = simulate_drop(fruits, 4, x)
    assert not is_lost(after)


def test_still_drops_when_every_candidate_is_lethal() -> None:
    """Return a move even on a stuck board (every drop crosses the losing line).

    The board is move 214 of seed=221700, the position where the side with the lethal-move filter actually died.
    If the candidates were emptied when there are 0 surviving candidates, the policy could not return a move.
    """
    raw = (
        (1, 22.6, 38.4), (1, 172.2, 42.3), (4, 261.7, 65.7), (5, 109.3, 73.0),
        (5, 348.8, 75.7), (2, 198.7, 81.2), (3, 31.8, 88.0), (3, 252.4, 133.4),
        (2, 28.3, 144.0), (6, 170.6, 158.3), (7, 330.7, 190.8), (7, 69.3, 228.3),
        (5, 228.0, 246.1), (1, 158.4, 252.8), (8, 321.6, 346.4), (9, 151.5, 368.4),
        (0, 16.1, 427.1), (4, 300.7, 459.4), (3, 31.7, 468.2), (3, 368.3, 468.3),
        (2, 237.0, 471.6), (2, 87.7, 471.7), (1, 190.3, 477.4), (0, 128.0, 483.9),
    )
    fruits = tuple(
        Fruit(type=t, x=x, y=y, radius=fruit_radius(t), confidence=90) for t, x, y in raw
    )
    grape_r = fruit_radius(2)
    xs = [clamp_drop_x(x, 2) for x in _candidates(list(fruits), 2, grape_r, extra_type=1)]
    # Premise: not a single surviving candidate. If this breaks, the test is not looking at a stuck board.
    assert all(is_lost(simulate_drop(fruits, 2, x)[0]) for x in xs)

    x = choose_x(_obs(held_type=2, fruits=fruits, next_type=1))
    assert grape_r <= x <= NORMALIZED_WIDTH - grape_r


def test_center_tiebreak_never_outranks_a_merge() -> None:
    """A term only for ordering. It cannot overturn even the cheapest merge (two cherries = 1 point).

    If this breaks, a term meant only to order the band starts deciding
    how good moves are. That is why bumpiness was retired (NOTES 'Retired: bumpiness (height variance)').
    """
    worst = max(center_tiebreak(0.0), center_tiebreak(NORMALIZED_WIDTH))
    assert worst < merge_points(0)


def test_center_tiebreak_prefers_the_middle() -> None:
    """Heavier toward the edges. Left-right symmetric, so independent of the board's size direction."""
    mid = NORMALIZED_WIDTH / 2
    assert center_tiebreak(mid) == 0.0
    assert center_tiebreak(mid + 40) < center_tiebreak(mid + 120)
    assert center_tiebreak(mid - 80) == center_tiebreak(mid + 80)
