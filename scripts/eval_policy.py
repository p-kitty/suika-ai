"""bootstrap / learned を同じ条件で sim 評価する。

用法:
  python scripts/eval_policy.py
  python scripts/eval_policy.py --policy learned --max-steps 100 --episodes 20
  python scripts/eval_policy.py --policy bootstrap --max-steps 100
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import LinearPolicy
from src.policy import choose_x
from src.reward import watermelon_count
from src.sim_env import SimEnv

DEFAULT_CKPT = ROOT / "artifacts" / "policy_sim.npz"


def run_episode(
    seed: int,
    *,
    choose,
    max_steps: int,
) -> dict[str, float]:
    env = SimEnv(seed=seed)
    obs = env.reset()
    total_reward = 0.0
    merges = 0
    steps = 0
    max_type = -1
    max_wm = 0
    for _ in range(max_steps):
        result = env.step(choose(obs))
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
        "win": 1.0 if result.info == "win" else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("bootstrap", "learned"),
        default="learned",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    if args.policy == "bootstrap":
        choose = choose_x
    else:
        if not args.checkpoint.is_file():
            raise SystemExit(f"checkpoint が無い: {args.checkpoint}")
        policy = LinearPolicy()
        policy.load(args.checkpoint)

        def choose(obs):
            _, x, _ = policy.act(obs, greedy=True)
            return x

    rows = [
        run_episode(args.seed + i, choose=choose, max_steps=args.max_steps)
        for i in range(args.episodes)
    ]
    steps = [r["steps"] for r in rows]
    rewards = [r["reward"] for r in rows]
    merges = [r["merges"] for r in rows]
    max_types = [r["max_type"] for r in rows]
    print(f"policy={args.policy}  episodes={args.episodes}  max_steps={args.max_steps}")
    print(f"steps  mean={statistics.mean(steps):.1f}  median={statistics.median(steps):.1f}")
    print(f"reward mean={statistics.mean(rewards):.2f}")
    print(f"merges mean={statistics.mean(merges):.1f}")
    print(f"max_type mean={statistics.mean(max_types):.2f}  best={max(max_types):.0f}")
    print(f"double_wm episodes={sum(1 for r in rows if r['max_wm'] >= 2)}")
    print(f"win episodes={sum(1 for r in rows if r['win'] >= 1)}")


if __name__ == "__main__":
    main()
