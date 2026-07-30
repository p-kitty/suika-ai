"""線形方策の単体テスト。"""

import numpy as np

from src.agent import N_ACTIONS, LinearPolicy, action_to_x
from src.encode import OBS_DIM
from src.sim_env import SimEnv


def test_action_to_x_in_board() -> None:
    x0 = action_to_x(0, 0)
    x_last = action_to_x(N_ACTIONS - 1, 0)
    assert 0 < x0 < x_last < 400


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
