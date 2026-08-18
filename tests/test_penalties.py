"""大小順の減点と、その谷免除 (`_is_nestled`) の単体テスト。

手の選び方は tests/test_policy.py。ここは減点規則そのものの意味と、
いま分かっている抜け穴を固定する。
"""

import pytest

from src.observe import Observation
from src.penalties import _is_nestled, _size_order_penalty
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


def test_nestled_fruit_is_dropped_from_size_order() -> None:
    """谷に入った実は、大小順が逆転していても数えられなくなる。

    どちらの盤もグレープ (2) がデコポン (3) の左＝逆転で、違うのは
    デコポンまでの距離だけ。谷とみなされた側だけ逆転ぶんの減点が消える。
    """
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)
    in_valley = [pear, grape, _on_floor(DEKOPON, 230.0)]
    too_far = [pear, grape, _on_floor(DEKOPON, 330.0)]

    assert _is_nestled(grape, in_valley)
    assert not _is_nestled(grape, too_far)
    assert _size_order_penalty(in_valley, LARGE_LEFT) < _size_order_penalty(
        too_far, LARGE_LEFT
    )


@pytest.mark.xfail(strict=True, reason="谷免除が、逆転した実を並べ直した盤より安くする")
def test_inversion_costs_more_than_the_correct_order() -> None:
    """同じ 3 個なら、逆転している盤の方が正しい順の盤より高く付くこと。

    梨・デコポン・グレープを同じ位置に置き、真ん中と右を入れ替えるだけ。
    逆転した側ではグレープが梨とデコポンの谷に入るので免除が掛かり、
    大小順の減点が正しい順の盤を下回る。
    """
    pear = _on_floor(PEAR, 70.0)
    ordered = [pear, _on_floor(DEKOPON, 170.0), _on_floor(GRAPE, 230.0)]
    inverted = [pear, _on_floor(GRAPE, 170.0), _on_floor(DEKOPON, 230.0)]

    assert _size_order_penalty(inverted, LARGE_LEFT) > _size_order_penalty(
        ordered, LARGE_LEFT
    )


@pytest.mark.xfail(strict=True, reason="落とす実自身が谷の壁になり、作った逆転が無料になる")
def test_drop_does_not_exempt_the_inversion_it_creates() -> None:
    """seed=49140 の 9 手目。デコポンをグレープの小さい側へ置かないこと。

    落下前の盤でグレープは谷に入っていない (右に大きい実が無い)。
    デコポンをグレープの右へ置いた瞬間に梨とデコポンの谷ができ、
    そのデコポンが作った逆転が、そのデコポン自身のおかげで免除される。
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
