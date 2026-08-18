"""大小順 (横・縦) の減点と、その谷免除・段階ゲートの単体テスト。

手の選び方は tests/test_policy.py。ここは減点規則そのものの意味を固定する。
谷の判定 (`_is_nestled`) と、そのうち実際に免除する条件 (`_size_order_exempt`)
は別物なので、それぞれ分けて置く。

縦の大小順は「上が大きいときだけ掛かる」ことが要で、大きい実の肩に小さい実を
載せる形 (梯子) は型差がいくつあっても 0 でなければならない。そこを外すと、
盤を整える手そのものを減点しにいく。
"""

import math

from src.observe import Observation
from src.penalties import (
    VERTICAL_ORDER_WEIGHT,
    _is_nestled,
    _size_order_exempt,
    _size_order_penalty,
    _vertical_order_penalty,
    board_is_broken,
    inversion_fraction,
)
from src.policy import choose_x
from src.sim.sim_physics import simulate_drop_held
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit

PEAR, DEKOPON, GRAPE = 6, 3, 2
MELON, ORANGE, CHERRY, PEACH, APPLE = 9, 4, 0, 7, 5
# 盤面の大小の向き。+1 = 左が大きい。
LARGE_LEFT = 1


def _on_floor(fruit_type: int, x: float) -> Fruit:
    radius = fruit_radius(fruit_type)
    return Fruit(
        type=fruit_type, x=x, y=NORMALIZED_HEIGHT - radius, radius=radius, confidence=90
    )


def test_nestled_needs_a_bigger_fruit_on_both_sides() -> None:
    """谷とみなすのは、左右とも自分より大きい実で挟まれたときだけ。"""
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)

    assert not _is_nestled(grape, [pear, grape])
    assert _is_nestled(grape, [pear, grape, _on_floor(DEKOPON, 230.0)])


def test_nestled_only_when_the_valley_is_narrow() -> None:
    """左右が離れていれば谷ではない。同じ 3 個でも間隔だけで判定が変わる。"""
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)

    assert not _is_nestled(grape, [pear, grape, _on_floor(DEKOPON, 330.0)])


def test_valley_fruit_is_exempt_only_with_a_merge_partner() -> None:
    """谷にいる実を大小順から外すのは、盤に同種の相方が残っているときだけ。

    どちらの盤もグレープ (2) がデコポン (3) の左＝逆転で、谷の形も同じ。
    違いは合体相手のグレープがもう 1 個あるかどうかだけ。相方がいなければ
    その谷から出る当てが無いので、ただの並び順違反として数える。
    """
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)
    dekopon = _on_floor(DEKOPON, 230.0)
    alone = [pear, grape, dekopon]
    with_partner = [pear, grape, dekopon, _on_floor(GRAPE, 300.0)]

    assert _is_nestled(grape, alone)
    assert _is_nestled(grape, with_partner)
    assert not _size_order_exempt(grape, alone)
    assert _size_order_exempt(grape, with_partner)


def test_inversion_costs_more_than_the_correct_order() -> None:
    """同じ 3 個なら、逆転している盤の方が正しい順の盤より高く付くこと。

    梨・デコポン・グレープを同じ位置に置き、真ん中と右を入れ替えるだけ。
    逆転した側ではグレープが梨とデコポンの谷に入るので、免除の条件が
    緩いと大小順の減点が正しい順の盤を下回る。
    """
    pear = _on_floor(PEAR, 70.0)
    ordered = [pear, _on_floor(DEKOPON, 170.0), _on_floor(GRAPE, 230.0)]
    inverted = [pear, _on_floor(GRAPE, 170.0), _on_floor(DEKOPON, 230.0)]

    assert _size_order_penalty(inverted, LARGE_LEFT) > _size_order_penalty(
        ordered, LARGE_LEFT
    )


def test_drop_does_not_exempt_the_inversion_it_creates() -> None:
    """seed=49140 の 9 手目。デコポンをグレープの小さい側へ置かないこと。

    落下前の盤でグレープは谷に入っていない (右に大きい実が無い)。
    デコポンをグレープの右へ置くと梨とデコポンの谷ができるので、免除の条件が
    緩いと、そのデコポンが作った逆転がそのデコポン自身のおかげで消える。
    """
    fruits = (_on_floor(PEAR, 96.5), _on_floor(GRAPE, 207.4))
    assert not _is_nestled(fruits[1], list(fruits))

    obs = Observation(
        ready=True,
        blocked=False,
        fruits=fruits,
        held_type=DEKOPON,
        held_x=NORMALIZED_WIDTH / 2,
        next_type=0,
    )
    after, _merges, _types, _held_merged = simulate_drop_held(
        list(fruits), DEKOPON, choose_x(obs)
    )
    dekopon = next(f for f in after if f.type == DEKOPON)
    grape = next(f for f in after if f.type == GRAPE)

    assert dekopon.x < grape.x


# --- 縦の大小順 ---------------------------------------------------------


def _stacked(lower_type: int, upper_type: int, x: float = 200.0) -> list[Fruit]:
    """真上に積んだ 2 個。"""
    lo_r, up_r = fruit_radius(lower_type), fruit_radius(upper_type)
    lower = _on_floor(lower_type, x)
    upper = Fruit(
        type=upper_type, x=x, y=lower.y - lo_r - up_r, radius=up_r, confidence=90
    )
    return [lower, upper]


def _rest_on(a: Fruit, b: Fruit, fruit_type: int) -> Fruit:
    """a と b の両方に接して上に乗る実。肩の形を組む足場。"""
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


def test_vertical_order_ignores_a_small_fruit_on_a_big_one() -> None:
    """大きい実の上に小さい実は正しい向き。型差が開いていても 0。

    メロン (9) の上のオレンジ (4) は型差 5 だが、縦の並びとしては正しい。
    ここを型差だけで数えると、盤を整える手を減点することになる。
    """
    assert _vertical_order_penalty(_stacked(MELON, ORANGE)) == 0.0
    assert _vertical_order_penalty(_stacked(PEACH, CHERRY)) == 0.0


def test_vertical_order_charges_a_big_fruit_on_a_small_one() -> None:
    """小さい実の上に大きい実を乗せたら型差ぶん減点。"""
    assert _vertical_order_penalty(_stacked(ORANGE, MELON)) == 5 * VERTICAL_ORDER_WEIGHT
    assert _vertical_order_penalty(_stacked(CHERRY, PEACH)) == 7 * VERTICAL_ORDER_WEIGHT


def test_vertical_order_ignores_fruits_merely_side_by_side() -> None:
    """横に並んだだけの組は「積んである」ではない。高さが同じなら 0。"""
    row = [_on_floor(CHERRY, 100.0), _on_floor(MELON, 300.0)]
    assert _vertical_order_penalty(row) == 0.0


def test_vertical_order_leaves_the_ladder_alone() -> None:
    """梯子の肩は素通りする。角桃の内側に梨、その肩にリンゴ・オレンジ。

    これがまさに「大きい側で発火を待つ」形なので、縦の規則がここに掛かると
    組み立てを自分で潰す。型差 (桃 7 とオレンジ 4 で 3) があっても 0。
    """
    peach_r, pear_r = fruit_radius(PEACH), fruit_radius(PEAR)
    peach = _on_floor(PEACH, peach_r + 2)
    pear = Fruit(
        type=PEAR,
        x=peach.x + peach_r + pear_r,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    apple = _rest_on(peach, pear, APPLE)
    orange = _rest_on(apple, pear, ORANGE)

    assert _vertical_order_penalty([peach, pear, apple, orange]) == 0.0


# --- 段階のゲート -------------------------------------------------------


def test_inversion_fraction_reads_the_board_order() -> None:
    """整列した盤は 0、逆に並べた盤は 1。sign は呼び元が渡す。"""
    ordered = [_on_floor(PEAR, 70.0), _on_floor(DEKOPON, 200.0), _on_floor(GRAPE, 320.0)]
    assert inversion_fraction(ordered, 1) == 0.0
    assert inversion_fraction(ordered, -1) == 1.0


def test_board_is_broken_only_past_the_threshold() -> None:
    """整った盤では立て直し側の規則を掛けない。

    ゲートの向きだけを固定する。しきい値そのものは A/B で決める数字。
    """
    ordered = [_on_floor(PEAR, 70.0), _on_floor(DEKOPON, 200.0), _on_floor(GRAPE, 320.0)]
    assert not board_is_broken(ordered, 1)
    assert board_is_broken(ordered, -1)
