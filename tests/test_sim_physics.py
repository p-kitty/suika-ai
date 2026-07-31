"""落下・転がり・衝突押し・合成の単体テスト。方策の手選びは見ない。"""

import math

from src.sim_physics import land_y, preview_land, simulate_drop
from src.vision.classify import fruit_radius
from src.vision.colors import MAX_FRUIT_TYPE
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit


def test_land_y_on_floor_when_empty() -> None:
    held_r = fruit_radius(0)
    assert abs(land_y((), 200, held_r) - (NORMALIZED_HEIGHT - held_r)) < 1e-6


def test_land_y_rests_on_fruit() -> None:
    held_r = fruit_radius(0)
    fruit = Fruit(type=1, x=200, y=400, radius=20, confidence=90)
    land = land_y((fruit,), 200, held_r)
    assert abs(land - (fruit.y - fruit.radius - held_r)) < 1e-6


def test_drop_on_slope_rolls_to_floor() -> None:
    pear_r = fruit_radius(6)
    cherry_r = fruit_radius(0)
    pear = Fruit(type=6, x=200, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    drop_x = pear.x + pear_r * 0.55
    land_x, land_y_ = preview_land((pear,), 0, drop_x, cherry_r)
    assert land_x > drop_x
    assert land_y_ >= NORMALIZED_HEIGHT - cherry_r - 1.0
    assert land_x >= pear.x + pear_r + cherry_r - 2.0


def test_same_type_floor_contact_merges() -> None:
    # 床で同種が触れたら、押しで離さず合体する。
    cherry_r = fruit_radius(0)
    a = Fruit(
        type=0,
        x=200,
        y=NORMALIZED_HEIGHT - cherry_r,
        radius=cherry_r,
        confidence=90,
    )
    after, merges, _types = simulate_drop((a,), 0, a.x + 2 * cherry_r)
    assert merges >= 1
    assert any(f.type == 1 for f in after)


def test_large_fruit_rolls_off_instead_of_wedging_on_wall() -> None:
    # 大実は壁で中心が止まるが、支えをどかして床まで落ちる。
    pear_r = fruit_radius(6)
    melon_r = fruit_radius(9)
    pear = Fruit(
        type=6,
        x=200,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    after, merges, _types = simulate_drop((pear,), 9, pear.x + pear_r * 0.4)
    assert merges == 0
    melon = next(f for f in after if f.type == 9)
    assert abs(melon.y - (NORMALIZED_HEIGHT - melon_r)) < 2.0


def test_foreign_floor_hit_slides_toward_wall() -> None:
    # 斜面から異種へ床接触したら、押された実は壁近くまで滑る。
    pear_r = fruit_radius(6)
    orange_r = fruit_radius(4)
    pear = Fruit(
        type=6,
        x=100,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    orange = Fruit(
        type=4,
        x=250,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    drop_x = pear.x + pear_r * 0.55
    after, _merges, _types = simulate_drop((pear, orange), 0, drop_x)
    moved = [f for f in after if f.type == 4]
    assert moved
    wall = NORMALIZED_WIDTH - orange_r
    assert moved[0].x > orange.x + 40.0
    assert abs(moved[0].x - wall) < 3.0


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


def test_fruit_above_deleted_watermelon_falls() -> None:
    # スイカ同士の合成で支えが消えた実は空中に残さず床へ落とす。
    melon_r = fruit_radius(MAX_FRUIT_TYPE)
    cherry_r = fruit_radius(0)
    a = Fruit(
        type=MAX_FRUIT_TYPE,
        x=200,
        y=NORMALIZED_HEIGHT - melon_r,
        radius=melon_r,
        confidence=90,
    )
    b = Fruit(
        type=MAX_FRUIT_TYPE,
        x=200 + 2 * melon_r + 4,
        y=NORMALIZED_HEIGHT - melon_r,
        radius=melon_r,
        confidence=90,
    )
    dx = melon_r * 0.3
    cy = a.y - math.sqrt((melon_r + cherry_r) ** 2 - dx * dx)
    cherry = Fruit(type=0, x=a.x + dx, y=cy, radius=cherry_r, confidence=90)
    after, merges, _types = simulate_drop((a, b, cherry), MAX_FRUIT_TYPE, a.x)
    assert merges >= 1
    cherries = [f for f in after if f.type == 0]
    assert len(cherries) == 1
    assert abs(cherries[0].y - (NORMALIZED_HEIGHT - cherry_r)) < 2.0


def test_hit_from_left_pushes_existing_right() -> None:
    # 左肩に当たった既存実は右へはっきり動く。
    orange_r = fruit_radius(4)
    orange = Fruit(
        type=4,
        x=220,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    drop_x = orange.x - orange_r * 0.35
    after, _merges, _types = simulate_drop((orange,), 0, drop_x)
    moved = [f for f in after if f.type == 4]
    assert moved
    assert moved[0].x > orange.x + 8.0


def test_merge_from_right_pushes_neighbor_right() -> None:
    # 右から合成すると、右隣の実も右へはっきり動く。
    cherry_r = fruit_radius(0)
    grape_r = fruit_radius(2)
    floor_c = NORMALIZED_HEIGHT - cherry_r
    a = Fruit(type=0, x=180, y=floor_c, radius=cherry_r, confidence=90)
    b = Fruit(type=0, x=180 + 2 * cherry_r + 1, y=floor_c, radius=cherry_r, confidence=90)
    neigh = Fruit(
        type=2,
        x=b.x + cherry_r + grape_r,
        y=NORMALIZED_HEIGHT - grape_r,
        radius=grape_r,
        confidence=90,
    )
    after, merges, _types = simulate_drop((a, b, neigh), 0, b.x + cherry_r * 0.5)
    assert merges >= 1
    grapes = [f for f in after if f.type == 2]
    assert grapes
    assert grapes[0].x > neigh.x + 10.0


def test_left_shoulder_of_grape_rolls_to_left_floor() -> None:
    grape_r = fruit_radius(2)
    straw_r = fruit_radius(1)
    grape = Fruit(type=2, x=160, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    drop_x = grape.x - grape_r * 0.5
    land_x, land_y_ = preview_land((grape,), 1, drop_x, straw_r)
    assert land_x < grape.x - grape_r
    assert land_y_ >= NORMALIZED_HEIGHT - straw_r - 1.0


def test_foreign_center_drop_rolls_off() -> None:
    # 異種の真上は安定せず、左右どちらかの床へ転がる。
    apple_r = fruit_radius(5)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    land_x, land_y_ = preview_land((apple,), 4, apple.x, orange_r)
    assert land_y_ >= NORMALIZED_HEIGHT - orange_r - 1.0
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


def test_shallow_shoulder_of_same_type_still_merges() -> None:
    # 同種は肩でも触れたら合成する。
    apple_r = fruit_radius(5)
    melon_r = fruit_radius(9)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    melon = Fruit(
        type=9,
        x=200,
        y=apple.y - apple_r - melon_r,
        radius=melon_r,
        confidence=90,
    )
    fruits = (apple, melon)
    shallow = melon.x + melon_r * 0.72
    after, merges, _types = simulate_drop(fruits, 9, shallow)
    assert merges >= 1
    assert sum(1 for f in after if f.type == 9) <= 1


def test_blocked_column_misses_but_open_same_type_merges() -> None:
    # 真上を異種で塞がれた列は直撃では合成しない。空き側の同種なら合成できる。
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
    assert simulate_drop(fruits, 7, left.x)[1] == 0
    assert simulate_drop(fruits, 7, right.x)[1] >= 1


def test_merges_sandwiched_same_type_despite_foreign_slope() -> None:
    # 大実の谷に挟まった同種は、狙いが少しずれても異種斜面で逃げず合体する。
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
    for d in (-8.0, -4.0, 0.0, 4.0, 8.0):
        after, merges, _types = simulate_drop(fruits, 2, gx + d)
        assert merges >= 1, d
        assert any(f.type == 3 for f in after)
