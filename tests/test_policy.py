"""薄い bootstrap 方策の単体テスト。画面は使わない。

具体手順 (押し込み・連鎖隙間空け) は要求しない。
合成・危険回避・埋め込み・転がり事故・谷育成の手選びだけ固定する。
落下物理そのものは tests/test_sim_physics.py。
"""

import math

from src.observe import Observation
from src.penalties import (
    FOREIGN_AIM_CENTER_FRAC,
    FOREIGN_AIM_PENALTY,
    foreign_aim_penalty,
    ideal_x,
)
from src.policy import _score, choose_x
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
    """危険な山から離れて落ち着くこと。狙う列 (x) 自体は問わない (上と同じ理由)。"""
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
    # メイトへ寄せる手が、異種の真上に積む手より良い。
    assert _score(obs, mate.x, straw_r) > _score(obs, buried.x, straw_r)
    x = choose_x(obs)
    after, merges, _types = simulate_drop((buried, mate), 1, x)
    # 真上埋め込みは選ばない。かすりで転がってメイト合成するのは可。
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
    # 衝突で弾かれるので、狙った列 x ではなく実際の着地位置で判定する。
    after, _merges, _merge_types, held_merged = simulate_drop_held(fruits, 2, x)
    land_x, _land_y = landed_xy(fruits, after, 2, x, grape_r, held_merged)
    assert abs(land_x - target.x) < cherry_r + grape_r * 2 + 40


def test_small_fruit_goes_right_of_large() -> None:
    """大きい実から離れて落ち着くこと。狙う列 (x) 自体は問わない。

    大きい実のすぐ脇を狙っても転がって離れた位置に収まれば意図通り。
    狙点を NORMALIZED_WIDTH の絶対値と比べると、転がりの分だけ無関係な
    列を選んでも落ち続ける (実測: x=132 を狙っても x=384 まで転がる)。
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
    # 谷に ぶどう、held/next が いちご (谷の実のひとつ下)。2 枚落とせば ぶどうに
    # なって谷の ぶどう と合体するので、隅に逃がさず谷に置く。
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
    # 谷のゴミより小さい実を足して掃除不能にしない。
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
    # next が同種だと谷で合成待ちになり正当に高得点になるので、別種で見る。
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
    # 接触のめり込みで床より数 px 沈むことがある。真上積み (~441) ではないことだけ見る。
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
    # 床 Segment 半径 2px ぶん、中心は NORMALIZED_HEIGHT - r より上に止まる。
    assert land_y >= NORMALIZED_HEIGHT - straw_r - 3.0
    assert land_x > NORMALIZED_WIDTH * 0.5
    assert land_x < cherry.x
    above = NORMALIZED_WIDTH - straw_r
    above_land, _ = preview_land((cherry,), 1, above, straw_r)
    assert above_land < NORMALIZED_WIDTH * 0.25
    assert _score(obs, x, straw_r) > _score(obs, above, straw_r)


def test_prefers_open_same_type_when_column_blocked() -> None:
    # 真上が塞がれた同種より、空き側の同種を選ぶ。
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
    # grape が 2 個で合成待ち。その谷に大きい orange を挟んで塞がない。
    grape_r = fruit_radius(2)
    orange_r = fruit_radius(4)
    floor_y = NORMALIZED_HEIGHT - grape_r
    a = Fruit(type=2, x=150, y=floor_y, radius=grape_r, confidence=90)
    b = Fruit(type=2, x=150 + grape_r * 2 + 30, y=floor_y, radius=grape_r, confidence=90)
    valley = (a.x + b.x) / 2
    obs = _obs(held_type=4, fruits=(a, b))
    x = choose_x(obs)
    # 谷に落とす手より、選んだ手の方が良い。
    assert _score(obs, x, orange_r) > _score(obs, valley, orange_r)
    land_x, _land_y = preview_land((a, b), 4, x, orange_r)
    assert not (a.x < land_x < b.x)


def test_avoids_foreign_center_stack() -> None:
    # 異種のガチ真上より、空き床や隣を選ぶ。
    apple_r = fruit_radius(5)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    obs = _obs(held_type=4, fruits=(apple,))
    x = choose_x(obs)
    assert abs(x - apple.x) > apple_r * FOREIGN_AIM_CENTER_FRAC
    assert _score(obs, x, orange_r) > _score(obs, apple.x, orange_r)


def test_foreign_aim_ignores_buried_foreign() -> None:
    # 下の異種中央列でも、幾何の真下接触が上の実の肩なら減点しない。
    apple_r = fruit_radius(5)
    grape_r = fruit_radius(2)
    orange_r = fruit_radius(4)
    apple = Fruit(type=5, x=200, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    grape_x = apple.x + apple_r * 0.45
    grape_y = apple.y - math.sqrt((apple_r + grape_r) ** 2 - (grape_x - apple.x) ** 2)
    grape = Fruit(type=2, x=grape_x, y=grape_y, radius=grape_r, confidence=90)
    # りんご中央列を狙うが、真下接触するのはずれたぶどうの方。
    assert foreign_aim_penalty((apple, grape), apple.x, 4, orange_r) == 0.0
    # 異種の頭をガチ真上から狙えば減点する。
    assert foreign_aim_penalty((apple,), apple.x, 4, orange_r) == FOREIGN_AIM_PENALTY


def test_foreign_aim_ok_when_same_type_below() -> None:
    # 真下が同種なら合体待ちなので FOREIGN_AIM しない (merges 条件ではない)。
    orange_r = fruit_radius(4)
    floor = NORMALIZED_HEIGHT - orange_r
    mate = Fruit(type=4, x=200, y=floor, radius=orange_r, confidence=90)
    assert foreign_aim_penalty((mate,), mate.x, 4, orange_r) == 0.0


def test_foreign_aim_penalizes_foreign_below_even_if_near_same_type() -> None:
    # 真下は異種。横に同種がいても真上狙いは減点する。
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
    # 3 + 1 で合成が進み、cherry は減って上位の実が残る。
    assert sum(1 for f in after if f.type == 0) <= 2
    assert any(f.type >= 1 for f in after)
    # 端に捨てて 4 個目にする手より、合成する手が明らかに良い。
    far = 40.0
    assert _score(obs, x, cherry_r) > _score(obs, far, cherry_r) + 20.0


def test_biggest_prefers_edge_over_center() -> None:
    # 最大実の大側寄せは size-order / ideal に任せる。空盤の桃は左寄り。
    peach_r = fruit_radius(7)
    obs = _obs(held_type=7)
    x = choose_x(obs)
    assert x < NORMALIZED_WIDTH * 0.45
    center = NORMALIZED_WIDTH / 2
    assert _score(obs, x, peach_r) > _score(obs, center, peach_r)


def test_large_fruits_prefer_clustering() -> None:
    # 大きい実どうしは近接。左端の桃に対し、梨は遠方より隣を選ぶ。
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


def test_floor_packed_measures_the_gap_against_the_drawn_fruit() -> None:
    # 壁から壁まで繋がっている必要はない。そのツモが収まらない隙間なら埋まり扱い。
    from src.penalties import _floor_packed

    assert not _floor_packed((), 4)

    row: list[Fruit] = []
    cursor = 0.0
    for fruit_type in (7, 6, 5, 5, 4, 4):
        r = fruit_radius(fruit_type)
        row.append(_floor(fruit_type, cursor + r))
        cursor += 2 * r
    assert _floor_packed(row, 4)
    # 右側をごっそり抜くと穴が空く。
    assert not _floor_packed(row[:2], 4)

    # 隙間がツモ直径ちょうどなら埋まり、少しでも広ければ埋まりでない。
    left = _floor(7, fruit_radius(7))
    edge = _floor(4, NORMALIZED_WIDTH - fruit_radius(4))
    for drop_type in (3, 4):
        gap = fruit_radius(drop_type) * 2.0
        right_x = left.x + left.radius + gap + fruit_radius(7)
        assert _floor_packed([left, _floor(7, right_x), edge], drop_type)
        assert not _floor_packed([left, _floor(7, right_x + 2.0), edge], drop_type)

    # デコポンが素直に入る隙間を、オレンジ基準で「置き場が無い」と読まない。
    right_x = left.x + left.radius + fruit_radius(3) * 2.0 + 4.0 + fruit_radius(7)
    board = [left, _floor(7, right_x), edge]
    assert _floor_packed(board, 4)
    assert not _floor_packed(board, 3)


def test_small_side_room_ignores_gap_blocked_by_overhang() -> None:
    # 床の隙間が幾何的に広くても、上に屋根 (別の実) が渡してあれば入らない。
    # 幾何だけで判定すると誤って room ありにする (指摘を受けて物理確認に変更)。
    from src.penalties import _small_side_room_ok

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
    # 対照: 屋根が無ければ同じ隙間幅で room ありになる。
    assert _small_side_room_ok((peach,), 4, orange_r, 7, sign=1)


def _rest_on(a: Fruit, b: Fruit, fruit_type: int) -> Fruit:
    """a と b の両方に接して上に乗る type の実。梯子の肩を組むテスト足場。"""
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
    """角桃 + 内側に梨、その 2 つの肩にリンゴ・オレンジ (梯子の完成形)。"""
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
    # 桃→梨→リンゴ→オレンジ が全部段として拾える。
    from src.ladder import find_anchor, rungs
    from src.policy import _order_sign

    fruits = _ladder_board()
    sign = _order_sign(fruits)
    anchor = find_anchor(fruits, sign)
    assert anchor is not None and anchor.type == 7
    assert sorted(rungs(fruits, anchor, sign)) == [4, 5, 6, 7]


def test_ladder_ignores_vertical_tower() -> None:
    # 桃の真上に梨を積んだ形は梯子ではない (崩れる形なので段に数えない)。
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
    # 組み上がっていれば、いまの choose_x が x 全振りの最良と同点を取る。
    # 発火だけを誘導しても意味がないことの固定 (足すなら組み立て側)。
    from src.reward import merge_score

    fruits = _ladder_board()
    # 床を右端まで埋める。すかすかだと梨が押し出されて形が保たない。
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
    # 最大より大側端に小実を置くのは可だが、L 中心より下の角ポケットは避ける。
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
    # 桃より左・床置き (y > peach.y) = 大側の角ポケット (sign=+1)。
    pocket = Fruit(
        type=4,
        x=orange_r + 2,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    assert pocket.x < peach.x
    assert pocket.y > peach.y
    # 桃の肩 (端側だが y <= peach.y)。
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
    # 小側 (右) の床に小実があっても、大側レイアウトでは角ポケットにしない。
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
    # 大側角の床ポケットへ落とさない。
    assert not (land_x < peach.x and land_y > peach.y)


def test_leaves_room_for_missing_rung_between_neighbours() -> None:
    """間の型が抜けているペアは、そのぶんの隙間を空けて置く。

    初手 orange の真横に grape を寄せると、次に dekopon が来たとき置き場が
    無く、grape の外側へ回して 4-2-3 の並びになってしまう (実測)。
    """
    orange_r = fruit_radius(4)
    dekopon_r = fruit_radius(3)
    grape_r = fruit_radius(2)

    # 1 手目 orange、2 手目 grape (next は dekopon)。
    x1 = choose_x(_obs(held_type=4, fruits=(), next_type=2))
    board1, _m, _t = simulate_drop((), 4, x1)
    x2 = choose_x(_obs(held_type=2, fruits=tuple(board1), next_type=3))
    board2, _m, _t = simulate_drop(board1, 2, x2)

    orange = next(f for f in board2 if f.type == 4)
    grape = next(f for f in board2 if f.type == 2)
    gap = abs(orange.x - grape.x) - orange_r - grape_r
    # dekopon 1 個ぶんに近い隙間を残す (押し込みぶんの余裕を見て 8 割)。
    assert gap > dekopon_r * 2 * 0.8

    # 3 手目 dekopon を置いても大小順が崩れない。
    x3 = choose_x(_obs(held_type=3, fruits=tuple(board2)))
    board3, _m, _t = simulate_drop(board2, 3, x3)
    assert [f.type for f in sorted(board3, key=lambda f: f.x)] == [2, 3, 4]
