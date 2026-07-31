"""薄い bootstrap 方策の単体テスト。画面は使わない。

具体手順 (育成優先・押し込み・連鎖隙間) は要求しない。
合成・危険回避・埋め込み・転がり事故・隙間ゴミだけ固定する。
"""

from src.observe import Observation
from src.policy import (
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


def test_does_not_stuff_cherry_between_pear_and_apple() -> None:
    pear_r = fruit_radius(6)
    apple_r = fruit_radius(5)
    cherry_r = fruit_radius(0)
    pear = Fruit(type=6, x=80, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    apple = Fruit(
        type=5,
        x=80 + pear_r + apple_r + cherry_r * 2 + 10,
        y=NORMALIZED_HEIGHT - apple_r,
        radius=apple_r,
        confidence=90,
    )
    gap_x = (pear.x + apple.x) / 2
    x = choose_x(_obs(held_type=0, fruits=(pear, apple)))
    assert x > apple.x
    assert abs(x - gap_x) > apple_r


def test_does_not_stuff_cherry_in_orange_grape_valley() -> None:
    orange_r = fruit_radius(4)
    grape_r = fruit_radius(2)
    cherry_r = fruit_radius(0)
    orange = Fruit(type=4, x=180, y=NORMALIZED_HEIGHT - orange_r, radius=orange_r, confidence=90)
    grape = Fruit(
        type=2,
        x=180 + orange_r + grape_r - 5,
        y=NORMALIZED_HEIGHT - grape_r,
        radius=grape_r,
        confidence=90,
    )
    obs = _obs(held_type=0, fruits=(orange, grape))
    x = choose_x(obs)
    land_x, land_y = _preview_land((orange, grape), 0, x, cherry_r)
    assert land_x > grape.x
    assert not (orange.x < land_x < grape.x)
    assert land_y >= NORMALIZED_HEIGHT - cherry_r - 1.0
    valley = (orange.x + grape.x) / 2
    assert _score(obs, x, cherry_r) > _score(obs, valley, cherry_r)


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


def test_avoids_foreign_center_stack() -> None:
    # 異種の中央真上より、空き床や隣を選ぶ。
    apple_r = fruit_radius(5)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    obs = _obs(held_type=4, fruits=(apple,))
    x = choose_x(obs)
    assert abs(x - apple.x) > apple_r * 0.25
    assert _score(obs, x, orange_r) > _score(obs, apple.x, orange_r)
