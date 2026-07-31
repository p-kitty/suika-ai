"""オフライン sim の単体テスト。"""

from src.policy import choose_x
from src.reward import CLEAR_SCORE, WATERMELON
from src.sim_env import SimEnv
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT
from src.vision.state import Fruit


def test_reset_ready_with_held_next() -> None:
    env = SimEnv(seed=0)
    obs = env.reset()
    assert obs.ready
    assert obs.held_type is not None
    assert obs.next_type is not None
    assert obs.fruits == ()


def test_bootstrap_plays_several_steps() -> None:
    env = SimEnv(seed=1)
    obs = env.reset()
    steps = 0
    for _ in range(8):
        x = choose_x(obs)
        result = env.step(x)
        obs = result.observation
        steps += 1
        if result.done:
            break
    assert steps >= 3
    assert obs.held_type is not None


def test_merge_can_happen_in_sim() -> None:
    env = SimEnv(seed=2)
    env.reset()
    # 同種を同じ列に二連続で落とす。
    env.held_type = 0
    env.next_type = 0
    first = env.step(200.0)
    env.held_type = 0
    second = env.step(200.0)
    assert first.merges + second.merges >= 1 or any(f.type >= 1 for f in second.observation.fruits)


def test_double_watermelon_clear_wins() -> None:
    """盤上にスイカが2個ある状態で片方と合成して減ったら win。"""
    env = SimEnv(seed=3)
    env.reset()
    w_r = fruit_radius(WATERMELON)
    y = NORMALIZED_HEIGHT - w_r
    left = 150.0
    right = left + 2 * w_r + 30.0
    env.fruits = [
        Fruit(type=WATERMELON, x=left, y=y, radius=w_r, confidence=90),
        Fruit(type=WATERMELON, x=right, y=y, radius=w_r, confidence=90),
    ]
    env.held_type = WATERMELON
    env.next_type = 0
    result = env.step(left)
    assert result.info == "win"
    assert result.done
    assert result.reward == CLEAR_SCORE
