"""bootstrap / learned を同じ条件で sim 評価する。

用法:
  python scripts/eval_policy.py
  python scripts/eval_policy.py --policy learned --max-steps 100 --episodes 20
  python scripts/eval_policy.py --policy bootstrap --max-steps 100 --workers 8
"""

from __future__ import annotations

import argparse
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT, ensure_import_path

ensure_import_path()

from src.agent import LinearPolicy
from src.parallel import default_workers
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
    total_score = 0.0
    merges = 0
    steps = 0
    max_type = -1
    max_wm = 0
    info = "ok"
    for _ in range(max_steps):
        result = env.step(choose(obs))
        obs = result.observation
        total_score += result.score
        merges += result.merges
        steps += 1
        info = result.info
        if obs.fruits:
            max_type = max(max_type, max(f.type for f in obs.fruits))
        max_wm = max(max_wm, watermelon_count(obs))
        if result.done:
            break
    return {
        "steps": float(steps),
        "score": total_score,
        "merges": float(merges),
        "max_type": float(max_type),
        "max_wm": float(max_wm),
        "win": 1.0 if info == "win" else 0.0,
    }


def _run_bootstrap_episode(seed: int, max_steps: int) -> dict[str, float]:
    """ProcessPool 用。"""
    return run_episode(seed, choose=choose_x, max_steps=max_steps)


def _run_learned_episode(
    seed: int, max_steps: int, checkpoint: str
) -> dict[str, float]:
    """ProcessPool 用。ワーカー側で重みを読む。"""
    policy = LinearPolicy()
    policy.load(Path(checkpoint))

    def choose(obs):
        _, x, _ = policy.act(obs, greedy=True)
        return x

    return run_episode(seed, choose=choose, max_steps=max_steps)


def run_episodes(
    *,
    policy_name: str,
    episodes: int,
    seed: int,
    max_steps: int,
    checkpoint: Path,
    workers: int,
) -> list[dict[str, float]]:
    seeds = [seed + i for i in range(episodes)]
    if workers <= 1 or episodes <= 1:
        if policy_name == "bootstrap":
            return [_run_bootstrap_episode(s, max_steps) for s in seeds]
        ckpt = str(checkpoint)
        return [_run_learned_episode(s, max_steps, ckpt) for s in seeds]

    rows: list[dict[str, float] | None] = [None] * episodes
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if policy_name == "bootstrap":
            future_to_idx = {
                pool.submit(_run_bootstrap_episode, s, max_steps): i
                for i, s in enumerate(seeds)
            }
        else:
            ckpt = str(checkpoint)
            future_to_idx = {
                pool.submit(_run_learned_episode, s, max_steps, ckpt): i
                for i, s in enumerate(seeds)
            }
        for future in as_completed(future_to_idx):
            rows[future_to_idx[future]] = future.result()
    assert all(r is not None for r in rows)
    return [r for r in rows if r is not None]


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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="エピソード並列数 (省略時は論理コア/2、1 で直列)",
    )
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else default_workers()

    if args.policy == "learned" and not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint が無い: {args.checkpoint}")

    rows = run_episodes(
        policy_name=args.policy,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        checkpoint=args.checkpoint,
        workers=workers,
    )
    steps = [r["steps"] for r in rows]
    scores = [r["score"] for r in rows]
    merges = [r["merges"] for r in rows]
    max_types = [r["max_type"] for r in rows]
    print(
        f"policy={args.policy}  episodes={args.episodes}  "
        f"max_steps={args.max_steps}  workers={workers}"
    )
    print(f"steps  mean={statistics.mean(steps):.1f}  median={statistics.median(steps):.1f}")
    print(f"score  mean={statistics.mean(scores):.2f}")
    print(f"merges mean={statistics.mean(merges):.1f}")
    print(f"max_type mean={statistics.mean(max_types):.2f}  best={max(max_types):.0f}")
    print(f"double_wm episodes={sum(1 for r in rows if r['max_wm'] >= 2)}")
    print(f"win episodes={sum(1 for r in rows if r['win'] >= 1)}")


if __name__ == "__main__":
    main()
