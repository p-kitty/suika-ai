"""報酬の単体テスト。"""

from src.observe import Observation
from src.vision.classify import fruit_radius
from src.reward import (
    CLEAR_SCORE,
    CREATE_SCORE,
    WATERMELON,
    cleared_double_watermelon,
    is_game_over,
    merge_points,
    step_reward,
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
    before = _obs()
    after = _obs()
    assert step_reward(before, after, merges=0, done=True) == 0.0


def test_no_step_bonus_without_merge() -> None:
    before = _obs()
    after = _obs()
    assert step_reward(before, after, merges=0, done=False) == 0.0


def test_merge_reward_is_create_score() -> None:
    cherry_r = fruit_radius(0)
    straw_r = fruit_radius(1)
    before = _obs(
        (
            Fruit(type=0, x=100, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90),
        )
    )
    after = _obs(
        (
            Fruit(type=1, x=100, y=NORMALIZED_HEIGHT - straw_r, radius=straw_r, confidence=90),
        )
    )
    assert step_reward(before, after, merges=1, merge_types=(0,), done=False) == 1.0


def test_no_bonus_for_keeping_double() -> None:
    w_r = fruit_radius(WATERMELON)
    two = _obs(
        (
            Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
            Fruit(type=WATERMELON, x=280, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
        )
    )
    assert step_reward(two, two, merges=0, done=False) == 0.0


def test_watermelon_clear_score() -> None:
    w_r = fruit_radius(WATERMELON)
    two = _obs(
        (
            Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
            Fruit(type=WATERMELON, x=280, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
        )
    )
    empty = _obs()
    assert step_reward(two, empty, merges=1, merge_types=(WATERMELON,), done=False) == 65.0


def test_win_reward_is_clear_score_only() -> None:
    w_r = fruit_radius(WATERMELON)
    two = _obs(
        (
            Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
            Fruit(type=WATERMELON, x=280, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
        )
    )
    empty = _obs()
    assert cleared_double_watermelon(two, empty, merges=1)
    got = step_reward(
        two, empty, merges=1, merge_types=(WATERMELON,), done=True, win=True
    )
    assert got == 65.0


def test_single_watermelon_clear_is_not_win() -> None:
    w_r = fruit_radius(WATERMELON)
    one = _obs(
        (Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),)
    )
    empty = _obs()
    assert not cleared_double_watermelon(one, empty, merges=1)


def test_chain_sums_each_merge() -> None:
    before = _obs()
    after = _obs()
    # cherry→straw (1) + straw→grape (3)
    assert step_reward(before, after, merge_types=(0, 1), done=False) == 4.0


def test_game_over_by_crown() -> None:
    big_r = fruit_radius(5)
    safe = _obs((Fruit(type=5, x=100, y=200, radius=big_r, confidence=90),))
    dead = _obs((Fruit(type=5, x=100, y=30, radius=big_r, confidence=90),))
    assert not is_game_over(safe)
    assert is_game_over(dead)


def test_watermelon_count() -> None:
    w_r = fruit_radius(WATERMELON)
    two = _obs(
        (
            Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
            Fruit(type=WATERMELON, x=280, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
        )
    )
    assert watermelon_count(two) == 2
