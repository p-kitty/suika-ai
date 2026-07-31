"""薄い bootstrap 方策の単体テスト。画面は使わない。

具体手順 (押し込み・連鎖隙間空け) は要求しない。
合成・危険回避・埋め込み・転がり事故・谷育成だけ固定する。
"""

from src.observe import Observation
from src.policy import (
    BURY_BLOCK_WEIGHT,
    _ideal_x,
    _land_y,
    _preview_land,
    _score,
    choose_x,
    simulate_drop,
)
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


def test_land_y_on_floor_when_empty() -> None:
    held_r = fruit_radius(0)
    assert abs(_land_y((), 200, held_r) - (NORMALIZED_HEIGHT - held_r)) < 1e-6


def test_land_y_rests_on_fruit() -> None:
    held_r = fruit_radius(0)
    fruit = Fruit(type=1, x=200, y=400, radius=20, confidence=90)
    land = _land_y((fruit,), 200, held_r)
    assert abs(land - (fruit.y - fruit.radius - held_r)) < 1e-6


def test_prefers_merge_that_lowers_stack() -> None:
    cherry_r = fruit_radius(0)
    floor_y = NORMALIZED_HEIGHT - cherry_r
    a = Fruit(type=0, x=250, y=floor_y, radius=cherry_r, confidence=90)
    b = Fruit(type=0, x=250 + 2 * cherry_r + 8, y=floor_y, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(a, b)))
    assert x > 200
    after, merges, _types = simulate_drop((a, b), 0, x)
    assert merges >= 1
    assert abs(x - (a.x + b.x) / 2) < cherry_r * 4


def test_does_not_bury_same_type_under_different() -> None:
    cherry_r = fruit_radius(0)
    straw_r = fruit_radius(1)
    floor_cherry = NORMALIZED_HEIGHT - cherry_r
    floor_straw = NORMALIZED_HEIGHT - straw_r
    buried = Fruit(type=0, x=100, y=floor_cherry, radius=cherry_r, confidence=90)
    mate = Fruit(type=1, x=300, y=floor_straw, radius=straw_r, confidence=90)
    x = choose_x(_obs(held_type=1, fruits=(buried, mate)))
    assert abs(x - mate.x) < straw_r * 3
    assert x > 200


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


def test_grows_grape_in_pear_valley_when_next_matches() -> None:
    # 大きい実の谷に、next と同種を置いて間を育てる。
    pear_r = fruit_radius(6)
    grape_r = fruit_radius(2)
    left = Fruit(type=6, x=150, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    right = Fruit(
        type=6,
        x=150 + pear_r * 2 + grape_r * 1.5,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    gap_x = (left.x + right.x) / 2
    fruits = (left, right)
    obs = _obs(held_type=2, fruits=fruits, next_type=2)
    x = choose_x(obs)
    land_x, _land_y = _preview_land(fruits, 2, x, grape_r)
    assert left.x < land_x < right.x
    far = NORMALIZED_WIDTH - grape_r - 8
    assert _score(obs, x, grape_r) > _score(obs, far, grape_r)


def test_drop_on_slope_rolls_to_floor() -> None:
    pear_r = fruit_radius(6)
    cherry_r = fruit_radius(0)
    pear = Fruit(type=6, x=200, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    drop_x = pear.x + pear_r * 0.55
    land_x, land_y = _preview_land((pear,), 0, drop_x, cherry_r)
    assert land_x > drop_x
    assert land_y >= NORMALIZED_HEIGHT - cherry_r - 1.0
    assert land_x >= pear.x + pear_r + cherry_r - 2.0


def test_merge_result_settles_from_midpoint() -> None:
    cherry_r = fruit_radius(0)
    floor_y = NORMALIZED_HEIGHT - cherry_r
    a = Fruit(type=0, x=200, y=floor_y, radius=cherry_r, confidence=90)
    b = Fruit(type=0, x=200 + 2 * cherry_r + 4, y=floor_y, radius=cherry_r, confidence=90)
    mid = (a.x + b.x) / 2
    after, merges, _types = simulate_drop((a, b), 0, mid)
    assert merges >= 1
    grown = [f for f in after if f.type == 1]
    assert grown
    assert abs(grown[0].x - mid) < cherry_r * 2


def test_strawberry_does_not_roll_left_of_grape() -> None:
    grape_r = fruit_radius(2)
    straw_r = fruit_radius(1)
    grape = Fruit(type=2, x=160, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    obs = _obs(held_type=1, fruits=(grape,))
    x = choose_x(obs)
    land_x, _land = _preview_land((grape,), 1, x, straw_r)
    assert land_x >= grape.x - 1.0
    left_shoulder = grape.x - grape_r * 0.5
    assert _score(obs, left_shoulder, straw_r) < _score(obs, x, straw_r)


def test_left_shoulder_of_grape_rolls_to_left_floor() -> None:
    grape_r = fruit_radius(2)
    straw_r = fruit_radius(1)
    grape = Fruit(type=2, x=160, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    drop_x = grape.x - grape_r * 0.5
    land_x, land_y = _preview_land((grape,), 1, drop_x, straw_r)
    assert land_x < grape.x - grape_r
    assert land_y >= NORMALIZED_HEIGHT - straw_r - 1.0


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
    land_x, land_y = _preview_land((straw,), 2, x, grape_r)
    assert land_y >= NORMALIZED_HEIGHT - grape_r - 1.0
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
    land_x, land_y = _preview_land((cherry,), 1, x, straw_r)
    assert land_y >= NORMALIZED_HEIGHT - straw_r - 1.0
    assert land_x > NORMALIZED_WIDTH * 0.5
    assert land_x < cherry.x
    above = NORMALIZED_WIDTH - straw_r
    above_land, _ = _preview_land((cherry,), 1, above, straw_r)
    assert above_land < NORMALIZED_WIDTH * 0.25
    assert _score(obs, x, straw_r) > _score(obs, above, straw_r)


def test_foreign_center_drop_rolls_off() -> None:
    # 異種の真上は安定せず、左右どちらかの床へ転がる。
    apple_r = fruit_radius(5)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    land_x, land_y = _preview_land((apple,), 4, apple.x, orange_r)
    assert land_y >= NORMALIZED_HEIGHT - orange_r - 1.0
    assert abs(land_x - apple.x) >= apple_r + orange_r - 2.0


def test_same_type_center_drop_still_merges() -> None:
    cherry_r = fruit_radius(0)
    a = Fruit(
        type=0,
        x=200,
        y=NORMALIZED_HEIGHT - cherry_r,
        radius=cherry_r,
        confidence=90,
    )
    after, merges, _types = simulate_drop((a,), 0, a.x)
    assert merges >= 1
    assert any(f.type == 1 for f in after)


def test_merges_sandwiched_same_type_despite_foreign_slope() -> None:
    # 大実の谷に挟まった同種は、狙いが少しずれても異種斜面で逃げず合体する。
    import math

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
    # 実プレイの aim 誤差くらいずらしても合体できる。
    for d in (-8.0, -4.0, 0.0, 4.0, 8.0):
        after, merges, _types = simulate_drop(fruits, 2, gx + d)
        assert merges >= 1, d
        assert any(f.type == 3 for f in after)
    obs = _obs(held_type=2, fruits=fruits)
    x = choose_x(obs)
    assert simulate_drop(fruits, 2, x)[1] >= 1


def test_does_not_block_waiting_pair_with_bigger_fruit() -> None:
    # grape が 2 個で合成待ち。その谷に大きい orange を挟んで塞がない。
    grape_r = fruit_radius(2)
    orange_r = fruit_radius(4)
    floor_y = NORMALIZED_HEIGHT - grape_r
    a = Fruit(type=2, x=150, y=floor_y, radius=grape_r, confidence=90)
    b = Fruit(type=2, x=150 + grape_r * 2 + 30, y=floor_y, radius=grape_r, confidence=90)
    valley = (a.x + b.x) / 2
    obs = _obs(held_type=4, fruits=(a, b))
    x = choose_x(obs)
    land_x, land_y = _preview_land((a, b), 4, x, orange_r)
    assert land_y >= NORMALIZED_HEIGHT - orange_r - 1.0
    assert not (a.x < land_x < b.x)
    assert _score(obs, x, orange_r) > _score(obs, valley, orange_r) + BURY_BLOCK_WEIGHT


def test_avoids_foreign_center_stack() -> None:
    # 異種の中央真上より、空き床や隣を選ぶ。
    apple_r = fruit_radius(5)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    obs = _obs(held_type=4, fruits=(apple,))
    x = choose_x(obs)
    assert abs(x - apple.x) > apple_r * 0.3
    assert _score(obs, x, orange_r) > _score(obs, apple.x, orange_r)


def test_merges_when_three_same_type_waiting() -> None:
    # 同種が 3 個ある盤では、持っている同種で早めに合成する。
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
    # 3 + 1 → 1 合成で cherry は 2 以下、straw が 1。
    assert sum(1 for f in after if f.type == 0) <= 2
    assert any(f.type == 1 for f in after)
    # 端に捨てて 4 個目にする手より、合成する手が明らかに良い。
    far = 40.0
    assert _score(obs, x, cherry_r) > _score(obs, far, cherry_r) + 20.0
