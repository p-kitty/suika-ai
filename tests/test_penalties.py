"""大小順の減点と、その谷免除、合体の寄せ方の単体テスト。

手の選び方は tests/test_policy.py。ここは減点規則そのものの意味を固定する。
谷の判定 (`_is_nestled`) と、そのうち実際に免除する条件 (`_size_order_exempt`)
は別物なので、それぞれ分けて置く。
"""

from src.observe import Observation
from src.penalties import (
    BLOCKED_PARTNER_WEIGHT,
    MERGE_BIG_SIDE_SLACK_FRAC,
    blocked_partner_penalty,
    _perch_penalty,
    _is_nestled,
    _size_order_exempt,
    _size_order_penalty,
    merge_big_side_bonus,
)
from src.policy import choose_x
from src.sim.sim_physics import simulate_drop_held
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit

PEAR, DEKOPON, GRAPE, STRAW, CHERRY = 6, 3, 2, 1, 0
PEACH = 7
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
    """谷にいる実を大小順から外すのは、同じ谷に相方が残っているときだけ。

    どちらの盤もグレープ (2) がデコポン (3) の左＝逆転で、谷の形も同じ。
    違いは合体相手のグレープが谷の中にもう 1 個いるかどうかだけ。相方が
    いなければその谷から出る当てが無いので、ただの並び順違反として数える。
    """
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)
    dekopon = _on_floor(DEKOPON, 230.0)
    alone = [pear, grape, dekopon]
    with_partner = [pear, grape, _on_floor(GRAPE, 200.0), dekopon]

    assert _is_nestled(grape, alone)
    assert _is_nestled(grape, with_partner)
    assert not _size_order_exempt(grape, alone)
    assert _size_order_exempt(grape, with_partner)


def test_valley_fruit_is_not_exempt_by_a_partner_outside_the_valley() -> None:
    """谷の外の相方では免除しない。壁の大実に阻まれて会えないため。

    seed=834761 の 35 手目がこの形だった (ナシとパインの谷に残ったいちごが、
    反対端のいちごを根拠に免除されていた)。
    """
    pear = _on_floor(PEAR, 70.0)
    grape = _on_floor(GRAPE, 170.0)
    dekopon = _on_floor(DEKOPON, 230.0)
    outside = [pear, grape, dekopon, _on_floor(GRAPE, 330.0)]

    assert _is_nestled(grape, outside)
    assert not _size_order_exempt(grape, outside)


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
    after, _merges, _types, _held_merged, _held_fruit = simulate_drop_held(
        list(fruits), DEKOPON, choose_x(obs)
    )
    dekopon = next(f for f in after if f.type == DEKOPON)
    grape = next(f for f in after if f.type == GRAPE)

    assert dekopon.x < grape.x


def test_merge_big_side_bonus_follows_the_board_direction() -> None:
    """同じ移動でも、大側がどちらかで合否が反転する。"""
    held_r = fruit_radius(DEKOPON)
    moved_right = _on_floor(DEKOPON, 200.0 + held_r * (MERGE_BIG_SIDE_SLACK_FRAC + 0.1))

    assert merge_big_side_bonus(200.0, moved_right, held_r, -1) > 0.0
    assert merge_big_side_bonus(200.0, moved_right, held_r, LARGE_LEFT) == 0.0


def test_merge_big_side_bonus_ignores_a_shift_under_the_slack() -> None:
    """合体位置は 2 中心の中点なので、半径未満のずれは寄せたうちに入れない。"""
    held_r = fruit_radius(DEKOPON)
    slack = held_r * MERGE_BIG_SIDE_SLACK_FRAC

    assert merge_big_side_bonus(200.0, _on_floor(DEKOPON, 200.0 + slack - 1.0), held_r, -1) == 0.0
    assert merge_big_side_bonus(200.0, _on_floor(DEKOPON, 200.0 + slack + 1.0), held_r, -1) > 0.0


def test_merge_big_side_bonus_needs_a_surviving_fruit() -> None:
    """スイカまで育って消えたときは寄せ先が無い (held_fruit が None)。"""
    assert merge_big_side_bonus(200.0, None, fruit_radius(DEKOPON), -1) == 0.0


def test_perch_is_free_in_a_one_step_notch() -> None:
    """1 段上の壁の窪みは次の段。裸の上面に載せたときだけ肩乗りとして数える。

    小実は盤が大実で埋まるとどの肩でも型差が開くので、免除が無いと逃げ場が
    無くなり、方策は肩を避けて別の小実に屋根を掛ける側へ倒れる
    (seed=890270 の 72 手目)。
    """
    peach = _on_floor(PEACH, 78.0)
    top_y = peach.y - peach.radius - fruit_radius(GRAPE)
    grape = Fruit(type=GRAPE, x=120.0, y=top_y, radius=fruit_radius(GRAPE), confidence=90)
    wall = Fruit(type=DEKOPON, x=180.0, y=top_y, radius=fruit_radius(DEKOPON), confidence=90)

    assert _perch_penalty([peach, grape]) > 0.0
    assert _perch_penalty([peach, grape, wall]) == 0.0


def test_perch_still_counts_a_deep_valley() -> None:
    """型差の開いた谷は段ではなく罠。肩乗りとして数え続ける。

    `_perch_penalty` を入れる動機になった局面のさくらんぼも、りんごとオレンジの
    谷に載っていた。谷なら免除にするとその症例ごと消える。
    """
    pine = _on_floor(8, 150.0)
    top_y = pine.y - pine.radius - fruit_radius(CHERRY)
    cherry = Fruit(type=CHERRY, x=150.0, y=top_y, radius=fruit_radius(CHERRY), confidence=90)
    walls = [
        Fruit(type=t, x=150.0 + dx, y=top_y, radius=fruit_radius(t), confidence=90)
        for t, dx in ((5, -60.0), (4, 60.0))
    ]

    assert _perch_penalty([pine, cherry, *walls]) > 0.0


# --- 相方から遮られた着地 (`blocked_partner_penalty`) ---
# 見るのは落とした実自身が入った位置だけ。盤全体の到達可能性は数えない。


def test_blocked_partner_fires_when_a_bigger_fruit_walls_off_the_only_partner() -> None:
    """間に大実が立っていて相方に会えない着地には掛かる。"""
    held = _on_floor(CHERRY, 40.0)
    partner = _on_floor(CHERRY, 330.0)
    wall = _on_floor(PEACH, 185.0)

    assert blocked_partner_penalty([held, partner], held) == 0.0
    assert blocked_partner_penalty([held, partner, wall], held) == BLOCKED_PARTNER_WEIGHT


def test_blocked_partner_clears_when_any_partner_is_reachable() -> None:
    """相方が 1 個でも届くなら掛からない。全部遮られて初めて掛かる二値。"""
    held = _on_floor(CHERRY, 40.0)
    walled = _on_floor(CHERRY, 330.0)
    wall = _on_floor(PEACH, 185.0)
    near = _on_floor(CHERRY, 90.0)

    assert blocked_partner_penalty([held, walled, wall], held) == BLOCKED_PARTNER_WEIGHT
    assert blocked_partner_penalty([held, walled, wall, near], held) == 0.0


def test_blocked_partner_ignores_fruits_without_a_partner() -> None:
    """相方が盤にいない実は取り逃しではない (そちらは bury_lone の担当)。"""
    held = _on_floor(CHERRY, 40.0)
    wall = _on_floor(PEACH, 185.0)

    assert blocked_partner_penalty([held, wall], held) == 0.0


def test_blocked_partner_does_not_count_walls_it_can_get_past() -> None:
    """壁として数えるのは自分より大きく、頭が両者の中心より上に出た実だけ。"""
    held = _on_floor(GRAPE, 40.0)
    partner = _on_floor(GRAPE, 330.0)
    # 同型は押し出せるし合体で消えるので壁ではない。
    same = _on_floor(GRAPE, 185.0)
    assert blocked_partner_penalty([held, partner, same], held) == 0.0
    # 1 段上は次の段。相方が来れば合体して壁に追いつける。
    rung = _on_floor(DEKOPON, 185.0)
    assert blocked_partner_penalty([held, partner, rung], held) == 0.0
    # 大きくても頭が沈んでいれば乗り越えられる。
    sunk = Fruit(
        type=PEACH, x=185.0, y=NORMALIZED_HEIGHT + 60.0, radius=fruit_radius(PEACH), confidence=90
    )
    assert blocked_partner_penalty([held, partner, sunk], held) == 0.0
