"""報酬の単体テスト。"""

import math

from src.observe import Observation
from src.vision.classify import fruit_radius
from src.reward import (
    CLEAR_SCORE,
    CREATE_SCORE,
    WATERMELON,
    cleared_double_watermelon,
    is_corner_watermelon,
    is_game_over,
    is_lost,
    merge_points,
    merge_score,
    watermelon_count,
)
from src.vision.normalized import NORMALIZED_HEIGHT
from src.vision.state import Fruit


def _obs(fruits: tuple[Fruit, ...] = ()) -> Observation:
    return Observation(
        ready=True,
        blocked=False,
        fruits=fruits,
        held_type=0,
        held_x=200.0,
        next_type=0,
    )


def test_merge_points_match_game_table() -> None:
    assert CREATE_SCORE == (0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55)
    assert merge_points(0) == 1  # cherry → straw
    assert merge_points(1) == 3  # straw → grape
    assert merge_points(9) == 55  # melon → watermelon
    assert merge_points(WATERMELON) == CLEAR_SCORE == 65


def test_death_gives_no_penalty() -> None:
    assert merge_score(()) == 0.0


def test_no_step_bonus_without_merge() -> None:
    assert merge_score() == 0.0


def test_merge_score_is_create_score() -> None:
    assert merge_score((0,)) == 1.0


def test_no_bonus_for_keeping_double() -> None:
    w_r = fruit_radius(WATERMELON)
    two = _obs(
        (
            Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
            Fruit(type=WATERMELON, x=280, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
        )
    )
    assert not cleared_double_watermelon(two, two, merges=0)
    assert merge_score(()) == 0.0


def test_watermelon_clear_score() -> None:
    assert merge_score((WATERMELON,)) == 65.0


def test_win_score_is_clear_score_only() -> None:
    w_r = fruit_radius(WATERMELON)
    two = _obs(
        (
            Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
            Fruit(type=WATERMELON, x=280, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
        )
    )
    empty = _obs()
    assert cleared_double_watermelon(two, empty, merges=1)
    assert merge_score((WATERMELON,)) == 65.0


def test_single_watermelon_clear_is_not_win() -> None:
    w_r = fruit_radius(WATERMELON)
    one = _obs(
        (Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),)
    )
    empty = _obs()
    assert not cleared_double_watermelon(one, empty, merges=1)


def test_chain_sums_each_merge() -> None:
    # cherry→straw (1) + straw→grape (3)
    assert merge_score((0, 1)) == 4.0


def test_game_over_by_crown() -> None:
    big_r = fruit_radius(5)
    safe = _obs((Fruit(type=5, x=100, y=200, radius=big_r, confidence=90),))
    dead = _obs((Fruit(type=5, x=100, y=30, radius=big_r, confidence=90),))
    assert not is_game_over(safe)
    assert is_game_over(dead)


def test_is_lost_reads_a_plain_fruit_list() -> None:
    """落下後の盤 (Observation ではなく実の列) をそのまま判定できること。

    方策は候補ごとの after をこの形で持つので、ここが列を受けないと
    `policy.choose_x` の致死手フィルタが書けない。
    """
    big_r = fruit_radius(5)
    assert not is_lost([])
    assert not is_lost([Fruit(type=5, x=100, y=200, radius=big_r, confidence=90)])
    assert is_lost([Fruit(type=5, x=100, y=30, radius=big_r, confidence=90)])


def test_watermelon_count() -> None:
    w_r = fruit_radius(WATERMELON)
    two = _obs(
        (
            Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
            Fruit(type=WATERMELON, x=280, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
        )
    )
    assert watermelon_count(two) == 2


def _corner_board(corner_type: int | None) -> tuple[Fruit, ...]:
    """左下にスイカ、その右上にメロン。角に corner_type の実を挟む。

    角の実は壁・床・スイカのすべてに接する位置に置く。そのときスイカは
    x = r + 2*sqrt(R*r) まで押し出される (接点の幾何から出る)。さくらんぼは
    この値が R を下回る＝角の隙間に収まるので、スイカは壁に付いたまま。
    """
    big_r = fruit_radius(WATERMELON)
    melon_r = fruit_radius(WATERMELON - 1)
    fruits: list[Fruit] = []
    wm_x = big_r
    if corner_type is not None:
        small_r = fruit_radius(corner_type)
        wm_x = max(big_r, small_r + 2.0 * math.sqrt(big_r * small_r))
        fruits.append(
            Fruit(
                type=corner_type,
                x=small_r,
                y=NORMALIZED_HEIGHT - small_r,
                radius=small_r,
                confidence=90,
            )
        )
    watermelon = Fruit(
        type=WATERMELON,
        x=wm_x,
        y=NORMALIZED_HEIGHT - big_r,
        radius=big_r,
        confidence=90,
    )
    # 右上に乗るメロン。スイカに接する 45 度の位置。
    offset = (big_r + melon_r) / math.sqrt(2.0)
    melon = Fruit(
        type=WATERMELON - 1,
        x=watermelon.x + offset,
        y=watermelon.y - offset,
        radius=melon_r,
        confidence=90,
    )
    fruits.extend((watermelon, melon))
    return tuple(fruits)


def test_corner_watermelon_needs_the_wall_and_the_floor() -> None:
    assert is_corner_watermelon(_corner_board(None))

    middle = tuple(
        Fruit(type=f.type, x=f.x + 60.0, y=f.y, radius=f.radius, confidence=f.confidence)
        for f in _corner_board(None)
    )
    assert not is_corner_watermelon(middle)


def test_corner_watermelon_survives_a_cherry_or_strawberry_in_the_corner() -> None:
    """角に収まる小実なら、押し出されてもまだ角スイカ。"""
    assert is_corner_watermelon(_corner_board(0))
    assert is_corner_watermelon(_corner_board(1))


def test_corner_watermelon_is_lost_to_a_grape_in_the_corner() -> None:
    """ぶどうはスイカを壁から 23 押し出す。ここまで浮くと角ではない。"""
    assert not is_corner_watermelon(_corner_board(2))
