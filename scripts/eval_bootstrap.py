"""薄い bootstrap 方策を sim で評価する。

用法:
  python scripts/eval_bootstrap.py
  python scripts/eval_bootstrap.py --episodes 50
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy import choose_x
from src.reward import watermelon_count
from src.sim_env import SimEnv


def run_episode(seed: int, max_steps: int = 200) -> dict[str, float]:
    env = SimEnv(seed=seed)
    obs = env.reset()
    total_reward = 0.0
    merges = 0
    steps = 0
    max_type = -1
    max_wm = 0
    for _ in range(max_steps):
        result = env.step(choose_x(obs))
        obs = result.observation
        total_reward += result.reward
        merges += result.merges
        steps += 1
        if obs.fruits:
            max_type = max(max_type, max(f.type for f in obs.fruits))
        max_wm = max(max_wm, watermelon_count(obs))
        if result.done:
            break
    return {
        "steps": float(steps),
        "reward": total_reward,
        "merges": float(merges),
        "max_type": float(max_type),
        "max_wm": float(max_wm),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    # choose_x は候補×next で重いので既定は短め。
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()

    rows = [run_episode(args.seed + i, args.max_steps) for i in range(args.episodes)]
    steps = [r["steps"] for r in rows]
    rewards = [r["reward"] for r in rows]
    merges = [r["merges"] for r in rows]
    max_types = [r["max_type"] for r in rows]
    print(f"episodes={args.episodes}")
    print(f"steps  mean={statistics.mean(steps):.1f}  median={statistics.median(steps):.1f}")
    print(f"reward mean={statistics.mean(rewards):.2f}")
    print(f"merges mean={statistics.mean(merges):.1f}")
    print(f"max_type mean={statistics.mean(max_types):.2f}  best={max(max_types):.0f}")
    print(f"double_wm episodes={sum(1 for r in rows if r['max_wm'] >= 2)}")


if __name__ == "__main__":
    main()
