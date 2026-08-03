"""Unit tests for the thin bootstrap policy. No screen is used.

Concrete procedures (push-ins, cascade gap opening) are not required.
Only move selection for merging, danger avoidance, burying, rolling accidents and valley growing is pinned.
The fall physics itself is in tests/test_sim_physics.py.
"""

import math

from src.observe import Observation
from src.policy import (
    FOREIGN_AIM_CENTER_FRAC,
    FOREIGN_AIM_PENALTY,
    _foreign_aim_penalty,
    _ideal_x,
    _score,
    choose_x,
)
from src.sim_physics import preview_land, simulate_drop
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
    assert abs(x - _ideal_x(0)) < 40


def test_prefers_same_type_over_empty_low_column() -> None:
    cherry_r = fruit_radius(0)
    same = Fruit(type=0, x=280, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(same,)))
    assert abs(x - same.x) < cherry_r * 3
    assert x > 200


def test_avoids_dangerous_tall_stack() -> None:
    big_r = fruit_radius(5)
    tall = Fruit(type=5, x=80, y=60 + big_r, radius=big_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(tall,)))
    assert abs(x - tall.x) > 80
    assert x >= NORMALIZED_WIDTH / 2


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
    x = choose_x(_obs(held_type=2, fruits=(target, wall), next_type=0))
    assert x > NORMALIZED_WIDTH / 2
    assert abs(x - target.x) < cherry_r + grape_r * 2 + 40


def test_small_fruit_goes_right_of_large() -> None:
    big_r = fruit_radius(6)
    big = Fruit(type=6, x=90, y=NORMALIZED_HEIGHT - big_r, radius=big_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(big,)))
    assert x > NORMALIZED_WIDTH / 2


def test_prefers_held_that_enables_next_merge() -> None:
    cherry_r = fruit_radius(0)
    grape_r = fruit_radius(2)
    cherry = Fruit(type=0, x=310, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=2, fruits=(cherry,), next_type=0))
    assert x > NORMALIZED_WIDTH / 2
    assert abs(x - cherry.x) < cherry_r + grape_r * 2 + 50


def test_grows_apple_in_pear_valley_when_held_and_next_are_one_smaller() -> None:
    # When held/next are both fruits one smaller than the walls, it can land stably in the pear valley.
    from src.policy import _valley_grow_ok

    pear_r = fruit_radius(6)
    apple_r = fruit_radius(5)
    left = Fruit(type=6, x=150, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    right = Fruit(
        type=6,
        x=150 + pear_r * 2 + apple_r * 1.2,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    fruits = (left, right)
    obs = _obs(held_type=5, fruits=fruits, next_type=5)
    mid = (left.x + right.x) / 2
    land_x, _land_y = preview_land(fruits, 5, mid, apple_r)
    assert left.x < land_x < right.x
    assert _valley_grow_ok(fruits, land_x, 5, 5)
    far = NORMALIZED_WIDTH - apple_r - 8
    # A valley where the growing exemption applies does not lose badly to placing by the wall through wrong_side.
    assert _score(obs, mid, apple_r) > _score(obs, far, apple_r) - 20.0


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
        x=160 + pear_r * 2 + 10,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    gx = left.x + pear_r * 0.3
    dx = gx - left.x
    gy = left.y - math.sqrt((pear_r + grape_r) ** 2 - dx * dx)
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
    # Even in the center column of a different type below, no penalty if the landing is on the shoulder of the fruit above.
    apple_r = fruit_radius(5)
    grape_r = fruit_radius(2)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    grape_x = apple.x + apple_r * 0.45
    grape_y = apple.y - math.sqrt((apple_r + grape_r) ** 2 - (grape_x - apple.x) ** 2)
    grape = Fruit(type=2, x=grape_x, y=grape_y, radius=grape_r, confidence=90)
    # Drop at the apple's center column, but it sits on the offset grape.
    land_x = apple.x
    dx = abs(land_x - grape.x)
    land_y = grape_y - math.sqrt((grape_r + orange_r) ** 2 - dx * dx)
    assert _foreign_aim_penalty((apple, grape), land_x, land_y, 4, orange_r) == 0.0
    # Landing straight down onto the head of a different type is penalized.
    on_apple_y = apple.y - (apple_r + orange_r)
    assert _foreign_aim_penalty((apple,), apple.x, on_apple_y, 4, orange_r) == FOREIGN_AIM_PENALTY


def test_foreign_aim_ok_when_same_type_below() -> None:
    # If directly below is the same type it is waiting to merge, so no FOREIGN_AIM (not a merges condition).
    orange_r = fruit_radius(4)
    floor = NORMALIZED_HEIGHT - orange_r
    mate = Fruit(type=4, x=200, y=floor, radius=orange_r, confidence=90)
    land_y = mate.y - 2 * orange_r
    assert _foreign_aim_penalty((mate,), mate.x, land_y, 4, orange_r) == 0.0


def test_foreign_aim_penalizes_foreign_below_even_if_near_same_type() -> None:
    # Directly below is a different type. Even with a same type beside it, aiming directly above is penalized (the loophole of rolling into a merge).
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
    on_apple_y = apple.y - (apple_r + orange_r)
    assert _foreign_aim_penalty((apple, mate), apple.x, on_apple_y, 4, orange_r) == FOREIGN_AIM_PENALTY


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


def test_floor_packed_allows_gaps_up_to_an_orange() -> None:
    # It need not be connected from wall to wall. A gap an orange does not fit counts as filled.
    from src.policy import FLOOR_PACKED_GAP, _floor_packed

    assert not _floor_packed(())

    row: list[Fruit] = []
    cursor = 0.0
    for fruit_type in (7, 6, 5, 5, 4, 4):
        r = fruit_radius(fruit_type)
        row.append(_floor(fruit_type, cursor + r))
        cursor += 2 * r
    assert _floor_packed(row)
    # Removing the whole right side opens a hole.
    assert not _floor_packed(row[:3])

    # A gap exactly the orange's diameter is filled, and any wider is not.
    left = _floor(7, fruit_radius(7))
    right_x = left.x + left.radius + FLOOR_PACKED_GAP + fruit_radius(7)
    edge = _floor(4, NORMALIZED_WIDTH - fruit_radius(4))
    assert _floor_packed([left, _floor(7, right_x), edge])
    assert not _floor_packed([left, _floor(7, right_x + 2.0), edge])


def test_small_side_room_ignores_gap_blocked_by_overhang() -> None:
    # Even if a floor gap is geometrically wide, it does not fit when a roof (another fruit) spans above it.
    # Judging by geometry alone wrongly says there is room (changed to a physics check after it was pointed out).
    from src.policy import _small_side_room_ok

    peach_r = fruit_radius(7)
    peach = _floor(7, peach_r + 2)
    roof_r = fruit_radius(6)
    roof = Fruit(
        type=6,
        x=NORMALIZED_WIDTH - roof_r - 2,
        y=NORMALIZED_HEIGHT - roof_r - 120,
        radius=roof_r,
        confidence=90,
    )
    pillar_r = fruit_radius(0)
    pillar = _floor(0, roof.x - roof_r - pillar_r + 4)
    fruits = (peach, roof, pillar)

    orange_r = fruit_radius(4)
    assert not _small_side_room_ok(fruits, 4, orange_r, 7, sign=1)
    # Control: without the roof, the same gap width has room.
    assert _small_side_room_ok((peach,), 4, orange_r, 7, sign=1)


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
    from src.policy import _ladder_anchor, _ladder_rungs, _order_sign

    fruits = _ladder_board()
    sign = _order_sign(fruits)
    anchor = _ladder_anchor(fruits, sign)
    assert anchor is not None and anchor.type == 7
    assert sorted(_ladder_rungs(fruits, anchor, sign)) == [4, 5, 6, 7]


def test_ladder_ignores_vertical_tower() -> None:
    # A pear stacked directly on top of the peach is not a ladder (a shape that collapses, so not counted as a rung).
    from src.policy import _ladder_anchor, _ladder_rungs, _order_sign

    peach_r, pear_r = fruit_radius(7), fruit_radius(6)
    peach = Fruit(
        type=7, x=peach_r + 2, y=NORMALIZED_HEIGHT - peach_r, radius=peach_r, confidence=90
    )
    tower = Fruit(
        type=6, x=peach.x, y=peach.y - peach_r - pear_r, radius=pear_r, confidence=90
    )
    fruits = (peach, tower)
    sign = _order_sign(fruits)
    anchor = _ladder_anchor(fruits, sign)
    assert anchor is not None
    assert sorted(_ladder_rungs(fruits, anchor, sign)) == [7]


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
    from src.policy import _big_layout_penalty

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
