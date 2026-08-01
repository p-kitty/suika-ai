"""BC / REINFORCE と生徒評価。"""

from __future__ import annotations

import statistics

import numpy as np

from ..agent import LinearPolicy
from ..sim_env import SimEnv


def match_rate(
    policy: LinearPolicy,
    obs_buf: list[np.ndarray],
    act_buf: list[int],
    *,
    rng: np.random.Generator,
    sample: int = 512,
) -> float:
    """バッファ上の greedy 一致率。"""
    n = len(obs_buf)
    if n == 0:
        return 0.0
    take = min(sample, n)
    idx = rng.choice(n, size=take, replace=False)
    hits = 0
    for i in idx:
        if int(policy.probs(obs_buf[i]).argmax()) == act_buf[i]:
            hits += 1
    return hits / take


def train_bc_epoch(
    policy: LinearPolicy,
    obs_buf: list[np.ndarray],
    act_buf: list[int],
    *,
    lr: float,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """1 エポック BC。平均 NLL。"""
    n = len(obs_buf)
    if n == 0:
        return 0.0
    order = np.arange(n)
    rng.shuffle(order)
    losses: list[float] = []
    for start in range(0, n, batch_size):
        batch = order[start : start + batch_size]
        loss = policy.bc_update(
            [obs_buf[i] for i in batch],
            [act_buf[i] for i in batch],
            lr=lr,
        )
        losses.append(loss)
    return statistics.fmean(losses)


def eval_student(
    policy: LinearPolicy,
    *,
    seed: int,
    episodes: int,
    max_steps: int,
) -> tuple[float, float]:
    """生徒 greedy の平均 score・平均ステップ。"""
    scores: list[float] = []
    steps_list: list[int] = []
    for i in range(episodes):
        env = SimEnv(seed=seed + i)
        obs = env.reset()
        total_score = 0.0
        steps = 0
        for _ in range(max_steps):
            _, x, _ = policy.act(obs, greedy=True)
            result = env.step(x)
            total_score += result.score
            obs = result.observation
            steps += 1
            if result.done:
                break
        scores.append(total_score)
        steps_list.append(steps)
    return statistics.fmean(scores), statistics.fmean(steps_list)


def run_rl_episode(
    env: SimEnv,
    policy: LinearPolicy,
    *,
    max_steps: int,
    gamma: float,
    lr: float,
    entropy_coef: float = 0.0,
) -> tuple[float, int]:
    """1 エピソード REINFORCE (互換用)。run_rl_batch を推奨。"""
    total, steps, obs_list, actions, rewards = _rollout_rl_episode(
        env, policy, max_steps=max_steps
    )
    returns = _discounted_returns(rewards, gamma)
    baseline = statistics.fmean(returns) if returns else 0.0
    advantages = [r - baseline for r in returns]
    policy.update(obs_list, actions, advantages, lr=lr, entropy_coef=entropy_coef)
    return total, steps


def _collect_rl_episode(
    env: SimEnv,
    policy: LinearPolicy,
    *,
    max_steps: int,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    obs = env.reset()
    obs_list: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        action, x, vec = policy.act(obs)
        obs_list.append(vec)
        actions.append(action)
        result = env.step(x)
        rewards.append(result.score)
        obs = result.observation
        if result.done:
            break
    return obs_list, actions, rewards


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns: list[float] = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


def _normalize_advantages(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = statistics.fmean(values)
    if len(values) == 1:
        return [0.0]
    std = statistics.pstdev(values)
    if std < 1e-8:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _rollout_rl_episode(
    env: SimEnv,
    policy: LinearPolicy,
    *,
    max_steps: int,
) -> tuple[float, int, list[np.ndarray], list[int], list[float]]:
    obs_list, actions, rewards = _collect_rl_episode(env, policy, max_steps=max_steps)
    return float(sum(rewards)), len(rewards), obs_list, actions, rewards


def run_rl_batch(
    policy: LinearPolicy,
    *,
    seeds: list[int],
    max_steps: int,
    gamma: float,
    lr: float,
    entropy_coef: float = 0.0,
) -> tuple[float, float, float]:
    """複数エピソードをまとめて REINFORCE 更新 (分散低減)。

    各 step の discounted return をバッチ全体で標準化して advantage にする。
    """
    step_chunks: list[tuple[list[np.ndarray], list[int], list[float]]] = []
    ep_returns: list[float] = []
    ep_steps: list[int] = []
    flat_returns: list[float] = []

    for seed in seeds:
        env = SimEnv(seed=seed)
        total, steps, obs_list, actions, rewards = _rollout_rl_episode(
            env, policy, max_steps=max_steps
        )
        ep_returns.append(total)
        ep_steps.append(steps)
        returns = _discounted_returns(rewards, gamma)
        step_chunks.append((obs_list, actions, returns))
        flat_returns.extend(returns)

    norm_returns = _normalize_advantages(flat_returns)
    batch_obs: list[np.ndarray] = []
    batch_actions: list[int] = []
    batch_adv: list[float] = []
    idx = 0
    for obs_list, actions, returns in step_chunks:
        n = len(returns)
        batch_obs.extend(obs_list)
        batch_actions.extend(actions)
        batch_adv.extend(norm_returns[idx : idx + n])
        idx += n

    loss = policy.update(
        batch_obs,
        batch_actions,
        batch_adv,
        lr=lr,
        entropy_coef=entropy_coef,
    )
    return statistics.fmean(ep_returns), statistics.fmean(ep_steps), loss
