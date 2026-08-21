"""薄い bootstrap 方策の単体テスト。画面は使わない。

具体手順 (押し込み・連鎖隙間空け) は要求しない。
合成・危険回避・埋め込み・転がり事故・谷育成の手選びだけ固定する。
落下物理そのものは tests/test_sim_physics.py。
"""

import math

from src.observe import Observation, clamp_drop_x
from src.penalties import (
    FOREIGN_AIM_CENTER_FRAC,
    FOREIGN_AIM_PENALTY,
    MERGE_BIG_SIDE_BONUS,
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
    after, _merges, _merge_types, held_merged, _held_fruit = simulate_drop_held(fruits, 2, x)
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
    from src.penalties import _corner_pocket_penalty

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
    assert _corner_pocket_penalty((peach, pocket), sign=1) > _corner_pocket_penalty(
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
    assert _corner_pocket_penalty((peach, right_floor), sign=1) < 20

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


def test_does_not_perch_small_fruit_on_biggest() -> None:
    """大実の山の上に型差の大きい実を載せない。

    盤は seed=642746 の 97 手目 (NOTES「解決済み: 大実の肩に小実を載せる」)。
    どこへ置いても何かを埋める盤で、パインの山に載せる手だけが減点ゼロだった。
    狙う列は問わず、さくらんぼがパインの footprint の中でパインの中心より
    上に収まらないことだけを見る。
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
    after, _merges, _types, held_merged, _held_fruit = simulate_drop_held(fruits, 0, x)
    land_x, land_y = landed_xy(fruits, after, 0, x, cherry_r, held_merged)
    on_pine = abs(land_x - pine.x) <= pine.radius + cherry_r
    assert not (on_pine and land_y + cherry_r <= pine.y)


def test_does_not_kill_itself_when_a_surviving_drop_exists() -> None:
    """生きる手があるうちは、死ぬ手を選ばないこと。

    盤は seed=982108 の 172 手目 (NOTES「危険な高さの傾斜をフィルタに置き換えた」)。
    致死候補 2 本・生存候補 31 本で、致死手が eval で 23.7 勝っている局面。
    減点で死を表していた頃はここで左端の山にオレンジを重ねて自滅していた。
    狙う列は問わず、落ちたあとの盤が負けラインを越えないことだけを見る。
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
    """詰み盤 (どこへ落としても負けライン) でも手を返すこと。

    盤は seed=221700 の 214 手目、致死手フィルタを入れた側が実際に死んだ局面。
    生存候補が 0 本のとき候補を空にしてしまうと、方策が手を返せなくなる。
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
    # 前提: 生存候補が 1 本も無い。ここが崩れたら test は詰みを見ていない。
    assert all(is_lost(simulate_drop(fruits, 2, x)[0]) for x in xs)

    x = choose_x(_obs(held_type=2, fruits=fruits, next_type=1))
    assert grape_r <= x <= NORMALIZED_WIDTH - grape_r


def test_center_tiebreak_never_outranks_a_merge() -> None:
    """順位を決めるためだけの項。いちばん安い合成 (cherry 同士 = 1 点) すら覆せない。

    ここが破れると、帯の中で順序を付けるだけのつもりの項が手の良し悪しを
    決め始める。凸凹を廃止した理由がそれ (NOTES「廃止: 凸凹（高さの分散）」)。
    """
    worst = max(center_tiebreak(0.0), center_tiebreak(NORMALIZED_WIDTH))
    assert worst < merge_points(0)


def test_center_tiebreak_prefers_the_middle() -> None:
    """端へ寄るほど重い。左右対称なので盤の大小の向きに依らない。"""
    mid = NORMALIZED_WIDTH / 2
    assert center_tiebreak(mid) == 0.0
    assert center_tiebreak(mid + 40) < center_tiebreak(mid + 120)
    assert center_tiebreak(mid - 80) == center_tiebreak(mid + 80)


def test_merge_leaves_the_new_fruit_on_the_big_side() -> None:
    """同じ相方に当てる手でも、できた実が大側に残る当て方を選ぶ。

    合体した手は大小順を免除する (`_evaluate_drop`) ので、この項が無いと
    左右どちらから当てるかは `center_tiebreak` が決め、反動で新実が小側へ
    飛ぶ手がそのまま残る (seed=834761 の 18 手目)。
    """
    orange_r = fruit_radius(4)
    fruits = tuple(
        Fruit(
            type=t,
            x=x,
            y=NORMALIZED_HEIGHT - fruit_radius(t),
            radius=fruit_radius(t),
            confidence=90,
        )
        for t, x in ((0, 17.0), (1, 53.0), (4, 229.0), (7, 331.0))
    )
    partner = fruits[2]
    obs = _obs(held_type=4, fruits=fruits, next_type=2)

    x = choose_x(obs)
    after, merges, _types, held_merged, _held_fruit = simulate_drop_held(fruits, 4, x)
    assert merges == 1 and held_merged
    apple = next(f for f in after if f.type == 5)
    # 小側 (左) へ飛ばされていない。相方の居た所から自分の半径以上は戻らない。
    assert apple.x > partner.x - orange_r

    # 反動で左へ飛ばす当て方 (相方の左肩) は、同じ 1 合成でも下に付く。
    left_shoulder = partner.x - orange_r * 0.65
    flung, _m, _t, _h, _hx = simulate_drop_held(fruits, 4, left_shoulder)
    assert next(f for f in flung if f.type == 5).x < NORMALIZED_WIDTH * 0.4
    assert _score(obs, left_shoulder, orange_r) < _score(obs, x, orange_r)


def test_merge_big_side_bonus_never_outranks_a_merge() -> None:
    """寄せ方のボーナスは、いちばん安い合成 (cherry 同士 = 1 点) すら覆せない。

    合体どうしの順位は本家点で決まる、という性質をここで固定する
    (`center_tiebreak` の同名テストと同じ理由)。
    """
    assert MERGE_BIG_SIDE_BONUS < merge_points(0)


def test_prefers_a_big_shoulder_over_roofing_a_lone_fruit() -> None:
    """床が埋まった盤で、相方のいない実に屋根を掛けるより大実の肩を選ぶ。

    seed=214631 の 111 手目。右の山に載せるとさくらんぼ (相方なし) が
    オレンジで塞がれる。空いているのは左端のパインの肩だけで、そこは
    型差 4 なので `_perch_penalty` の対象外 (パインの肩はオレンジまで許す)。
    """
    fruits = tuple(
        Fruit(type=t, x=x, y=y, radius=fruit_radius(t), confidence=90)
        for t, x, y in (
            (8, 78.0, 421.0),
            (6, 165.0, 321.0),
            (7, 222.0, 431.0),
            (2, 308.0, 472.0),
            (6, 323.0, 361.0),
            (0, 384.0, 326.0),
        )
    )
    cherry = fruits[5]
    obs = _obs(held_type=4, fruits=fruits, next_type=0)

    x = choose_x(obs)
    _after, _m, _t, _h, orange = simulate_drop_held(fruits, 4, x)
    assert orange is not None
    assert orange.x < cherry.x - cherry.radius


def test_uses_the_next_rung_instead_of_roofing_a_small_fruit() -> None:
    """逃げ場が段の窪みにあるなら、小実に屋根を掛けずそこへ置く。

    seed=890270 の 72 手目。盤が大実で埋まっていて、グレープから見ると
    パインの肩は型差 6、桃の肩は 5。逃げ場が無いと屋根 (いちご 15) が
    いちばん安くなる。桃とデコポンの窪みは 1 段上の壁なので次の段。
    """
    fruits = tuple(
        Fruit(type=t, x=x, y=y, radius=fruit_radius(t), confidence=90)
        for t, x, y in (
            (8, 78.0, 422.0),
            (7, 222.0, 431.0),
            (3, 291.0, 362.0),
            (2, 314.0, 413.0),
            (1, 356.0, 392.0),
            (4, 359.0, 460.0),
            (0, 384.0, 413.0),
        )
    )
    straw = fruits[4]
    obs = _obs(held_type=2, fruits=fruits, next_type=2)

    x = choose_x(obs)
    _after, _m, _t, _h, grape = simulate_drop_held(fruits, 2, x)
    assert grape is not None
    # いちごの真上に屋根を掛けていない。
    assert abs(grape.x - straw.x) > straw.radius + grape.radius


def test_reaches_a_same_type_partner_under_a_roof() -> None:
    """屋根の下の同種の相方へ、横から転がして届く手を候補から落とさない。

    seed=871514 の 37 手目。さくらんぼはいちごの真下に埋まっていて、真上から
    落としてもいちごの肩に載るだけ。床まで抜けて相方に届く x は 2px 幅しか
    無く、そこを跨ぐと方策は代わりにりんご 2 個を梨にして (21 点)、
    さくらんぼを桃と梨の間に挟み込む。
    """
    fruits = tuple(
        Fruit(type=t, x=x, y=y, radius=fruit_radius(t), confidence=90)
        for t, x, y in (
            (7, 69.4, 430.6),
            (5, 184.6, 448.7),
            (4, 245.7, 385.8),
            (5, 306.8, 448.8),
            (1, 376.8, 449.8),
            (0, 383.9, 483.9),
        )
    )
    obs = _obs(held_type=0, fruits=fruits, next_type=4)

    x = choose_x(obs)
    after, _merges, _types, held_merged, _held = simulate_drop_held(fruits, 0, x)
    assert held_merged
    # 埋まっていた相方ごと片付いている (さくらんぼ 2 個 -> いちご -> ぶどう)。
    assert not [f for f in after if f.type == 0]
