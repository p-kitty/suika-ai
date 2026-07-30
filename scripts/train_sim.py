"""sim 上で線形方策を REINFORCE する。

用法:
  python scripts/train_sim.py
  python scripts/train_sim.py --episodes 200 --lr 0.02
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import LinearPolicy
from src.encode import encode
from src.sim_env import SimEnv


def run_episode(
    env: SimEnv,
    policy: LinearPolicy,
    *,
    max_steps: int,
    gamma: float,
) -> tuple[float, int, list[np.ndarray], list[int], list[float]]:
    obs = env.reset()
    obs_list: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        action, x, _probs = policy.act(obs)
        obs_list.append(encode(obs))
        actions.append(action)
        result = env.step(x)
        rewards.append(result.reward)
        obs = result.observation
        if result.done:
            break

    # 割引リターン。
    returns: list[float] = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running

    total = float(sum(rewards))
    return total, len(rewards), obs_list, actions, returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    policy = LinearPolicy(rng)
    env = SimEnv(seed=args.seed + 1)

    window: list[float] = []
    window_steps: list[int] = []
    for ep in range(1, args.episodes + 1):
        # エピソードごとに seed をずらす。
        env = SimEnv(seed=args.seed + 1000 + ep)
        total, steps, obs_list, actions, returns = run_episode(
            env, policy, max_steps=args.max_steps, gamma=args.gamma
        )
        # ベースライン: エピソード平均リターン。
        baseline = statistics.fmean(returns) if returns else 0.0
        advantages = [r - baseline for r in returns]
        policy.update(obs_list, actions, advantages, lr=args.lr)

        window.append(total)
        window_steps.append(steps)
        if ep % args.log_every == 0 or ep == 1:
            print(
                f"ep={ep:4d}  "
                f"reward={statistics.fmean(window):7.2f}  "
                f"steps={statistics.fmean(window_steps):5.1f}"
            )
            window.clear()
            window_steps.clear()


if __name__ == "__main__":
    main()
