"""大小順の減点と、その谷免除、合体の寄せ方の単体テスト。

手の選び方は tests/test_policy.py。ここは減点規則そのものの意味を固定する。
谷の判定 (`_is_nestled`) と、そのうち実際に免除する条件 (`_size_order_exempt`)
は別物なので、それぞれ分けて置く。
"""

from src.observe import Observation
from src.penalties import (
    MERGE_BIG_SIDE_SLACK_FRAC,
    _is_nestled,
    _size_order_exempt,
    _size_order_penalty,
    merge_lands_big_side,
)
from src.policy import choose_x
from src.sim.sim_physics import simulate_drop_held
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit

PEAR, DEKOPON, GRAPE = 6, 3, 2
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
    after, _merges, _types, _held_merged, _held_x = simulate_drop_held(
        list(fruits), DEKOPON, choose_x(obs)
    )
    dekopon = next(f for f in after if f.type == DEKOPON)
    grape = next(f for f in after if f.type == GRAPE)

    assert dekopon.x < grape.x


def test_merge_lands_big_side_follows_the_board_direction() -> None:
    """同じ移動でも、大側がどちらかで合否が反転する。"""
    held_r = fruit_radius(DEKOPON)
    moved_right = 200.0 + held_r * (MERGE_BIG_SIDE_SLACK_FRAC + 0.1)

    assert merge_lands_big_side(200.0, moved_right, held_r, -1)
    assert not merge_lands_big_side(200.0, moved_right, held_r, LARGE_LEFT)


def test_merge_lands_big_side_ignores_a_shift_under_the_slack() -> None:
    """合体位置は 2 中心の中点なので、半径未満のずれは寄せたうちに入れない。"""
    held_r = fruit_radius(DEKOPON)
    slack = held_r * MERGE_BIG_SIDE_SLACK_FRAC

    assert not merge_lands_big_side(200.0, 200.0 + slack - 1.0, held_r, -1)
    assert merge_lands_big_side(200.0, 200.0 + slack + 1.0, held_r, -1)


def test_merge_lands_big_side_needs_a_surviving_fruit() -> None:
    """スイカまで育って消えたときは寄せ先が無い (held_x が None)。"""
    assert not merge_lands_big_side(200.0, None, fruit_radius(DEKOPON), -1)
