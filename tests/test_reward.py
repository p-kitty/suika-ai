"""報酬の単体テスト。"""

from src.observe import Observation
from src.vision.classify import fruit_radius
from src.reward import (
    DEATH_PENALTY,
    DOUBLE_WATERMELON_BONUS,
    WATERMELON,
    WATERMELON_BONUS,
    is_game_over,
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


def test_death_penalty_on_done() -> None:
    before = _obs()
    after = _obs()
    assert step_reward(before, after, merges=0, done=True) == DEATH_PENALTY


def test_merge_and_progress_reward() -> None:
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
    reward = step_reward(before, after, merges=1, done=False)
    assert reward > 0.05


def test_watermelon_and_double_bonus() -> None:
    w_r = fruit_radius(WATERMELON)
    one = _obs(
        (Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),)
    )
    two = _obs(
        (
            Fruit(type=WATERMELON, x=120, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
            Fruit(type=WATERMELON, x=280, y=NORMALIZED_HEIGHT - w_r, radius=w_r, confidence=90),
        )
    )
    assert watermelon_count(two) == 2
    got = step_reward(one, two, merges=0, done=False)
    assert got >= WATERMELON_BONUS + DOUBLE_WATERMELON_BONUS


def test_game_over_by_crown() -> None:
    big_r = fruit_radius(5)
    safe = _obs((Fruit(type=5, x=100, y=200, radius=big_r, confidence=90),))
    dead = _obs((Fruit(type=5, x=100, y=30, radius=big_r, confidence=90),))
    assert not is_game_over(safe)
    assert is_game_over(dead)
