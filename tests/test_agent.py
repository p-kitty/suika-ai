"""線形方策の単体テスト。"""

import numpy as np

from src.agent import (
    N_ACTIONS,
    LinearPolicy,
    action_to_x,
    teacher_action_target,
    x_to_action,
)
from src.encode import OBS_DIM, encode
from src.policy import choose_x
from src.sim_env import SimEnv


def test_action_to_x_in_board() -> None:
    x0 = action_to_x(0, 0)
    x_last = action_to_x(N_ACTIONS - 1, 0)
    assert 0 < x0 < x_last < 400


def test_x_to_action_roundtrip() -> None:
    for action in range(N_ACTIONS):
        assert x_to_action(action_to_x(action, 0)) == action


def test_act_and_update() -> None:
    rng = np.random.default_rng(0)
    policy = LinearPolicy(rng)
    env = SimEnv(seed=0)
    obs = env.reset()
    action, x, vec = policy.act(obs)
    assert 0 <= action < N_ACTIONS
    assert vec.shape == (OBS_DIM,)
    assert 0 < x < 400

    probs = policy.probs(vec)
    assert probs.shape == (N_ACTIONS,)
    assert abs(probs.sum() - 1.0) < 1e-6
    loss = policy.update([vec], [action], [1.0], lr=0.01)
    assert np.isfinite(loss)


def test_bc_raises_teacher_prob() -> None:
    rng = np.random.default_rng(1)
    policy = LinearPolicy(rng)
    env = SimEnv(seed=1)
    obs = env.reset()
    teacher_a = x_to_action(choose_x(obs))
    vec = encode(obs)
    before = float(policy.probs(vec)[teacher_a])
    for _ in range(40):
        policy.bc_update([vec], [teacher_a], lr=0.1)
    after = float(policy.probs(vec)[teacher_a])
    assert after > before


def test_save_load_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(2)
    policy = LinearPolicy(rng)
    path = tmp_path / "p.npz"
    policy.save(path)
    other = LinearPolicy(np.random.default_rng(3))
    other.load(path)
    vec = np.zeros(OBS_DIM, dtype=np.float32)
    vec[0] = 1.0
    assert np.allclose(policy.probs(vec), other.probs(vec))


def test_soft_bc_raises_near_teacher() -> None:
    rng = np.random.default_rng(4)
    policy = LinearPolicy(rng)
    env = SimEnv(seed=4)
    obs = env.reset()
    teacher_x = choose_x(obs)
    teacher_a = x_to_action(teacher_x)
    vec = encode(obs)
    target = teacher_action_target(teacher_x)
    assert abs(target.sum() - 1.0) < 1e-6
    before = float(policy.probs(vec)[teacher_a])
    for _ in range(40):
        policy.bc_update_dist([vec], [target], lr=0.1)
    after = float(policy.probs(vec)[teacher_a])
    assert after > before
