"""Unit tests for the offline sim."""

from src.policy import choose_x
from src.reward import CLEAR_SCORE, WATERMELON
from src.sim import sim_env
from src.sim.sim_env import SimEnv
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


def test_seed_is_concrete_and_replayable() -> None:
    """Even with the seed omitted a concrete value remains, and that value replays the same draw sequence."""
    env = SimEnv()
    assert isinstance(env.seed, int)
    drawn = [env.reset().held_type] + [env.step(200.0).observation.held_type for _ in range(4)]

    replay = SimEnv(seed=env.seed)
    again = [replay.reset().held_type] + [replay.step(200.0).observation.held_type for _ in range(4)]
    assert again == drawn


def test_aim_noise_keeps_the_draws_and_moves_the_drop(monkeypatch) -> None:
    """The aim error lands within its half-width and does not reshuffle the fruit sequence of the seed,
    so an A/B with the error on stays paired on the same draws."""
    exact = SimEnv(seed=11)
    exact.reset()
    exact_draws = [exact.step(200.0).observation.held_type for _ in range(6)]

    monkeypatch.setattr(sim_env, "AIM_NOISE_PX", 8.0)
    noisy = SimEnv(seed=11)
    noisy.reset()
    noisy.held_type = 0
    first = noisy.step(200.0)
    landed = first.observation.fruits[0].x
    assert 192.0 - 1.0 <= landed <= 208.0 + 1.0
    assert landed != 200.0

    noisy = SimEnv(seed=11)
    noisy.reset()
    assert [noisy.step(200.0).observation.held_type for _ in range(6)] == exact_draws


def test_explicit_seed_is_kept() -> None:
    assert SimEnv(seed=7).seed == 7


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
    # Drop the same type twice in a row in the same column.
    env.held_type = 0
    env.next_type = 0
    first = env.step(200.0)
    env.held_type = 0
    second = env.step(200.0)
    assert first.merges + second.merges >= 1 or any(f.type >= 1 for f in second.observation.fruits)


def test_double_watermelon_clear_wins() -> None:
    """win when, with two watermelons on the board, merging with one of them reduces the count."""
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
    assert result.score == CLEAR_SCORE
