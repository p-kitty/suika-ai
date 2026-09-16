"""Evaluate bootstrap / learned in the sim under the same conditions.

Usage:
  python scripts/eval_policy.py
  python scripts/eval_policy.py --policy learned --max-steps 100 --episodes 20
  python scripts/eval_policy.py --policy bootstrap --max-steps 100 --workers 8
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT
from scripts._episodes import play_episode

from src.training.agent import LinearPolicy
from src.util.parallel import resolve_workers
from src.policy import choose_x

DEFAULT_CKPT = ROOT / "artifacts" / "policy_sim.npz"


def _run_bootstrap_episode(seed: int, max_steps: int) -> dict[str, float]:
    """For ProcessPool."""
    return play_episode(seed, choose_x, max_steps)


def _run_learned_episode(
    seed: int, max_steps: int, checkpoint: str
) -> dict[str, float]:
    """For ProcessPool. Weights are loaded on the worker side."""
    policy = LinearPolicy()
    policy.load(Path(checkpoint))

    def choose(obs):
        _, x, _ = policy.act(obs, greedy=True)
        return x

    return play_episode(seed, choose, max_steps)


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
    job: Callable[..., dict[str, float]]
    if policy_name == "bootstrap":
        job, extra = _run_bootstrap_episode, ()
    else:
        job, extra = _run_learned_episode, (str(checkpoint),)

    if workers <= 1 or episodes <= 1:
        if policy_name == "bootstrap":
            # With nothing to parallelize per episode, parallelize choose_x candidate evaluation (simulate_drop)
            # over processes. learned runs no physics, so it is not included.
            with ProcessPoolExecutor() as move_pool:
                return [
                    play_episode(s, lambda obs: choose_x(obs, pool=move_pool), max_steps)
                    for s in seeds
                ]
        return [job(s, max_steps, *extra) for s in seeds]

    # Order by episode number, not completion order, to match a serial run.
    rows: list[dict[str, float] | None] = [None] * episodes
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(job, s, max_steps, *extra): i for i, s in enumerate(seeds)
        }
        for future in as_completed(future_to_idx):
            rows[future_to_idx[future]] = future.result()
    return [row for row in rows if row is not None]


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
        help="episode parallelism (logical cores/2 when omitted, 1 for serial)",
    )
    args = parser.parse_args()
    workers = resolve_workers(args.workers)

    if args.policy == "learned" and not args.checkpoint.is_file():
        raise SystemExit(f"no checkpoint: {args.checkpoint}")

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
    truncated = sum(1 for r in rows if r["truncated"])
    print(
        f"steps  mean={statistics.mean(steps):.1f}  "
        f"median={statistics.median(steps):.1f}  max={max(steps):.0f}  "
        f"truncated={truncated}/{len(rows)}"
    )
    # Setting the cap needs the tail, not the mean. Truncated runs are the long games, so
    # 'what fraction is cut at this cap' can be read straight from the quantiles (from a run with 0
    # truncations, the truncation rate for any cap comes out of this one run).
    tail = sorted(steps)
    marks = " ".join(
        f"p{int(q * 100)}={tail[min(len(tail) - 1, int(q * len(tail)))]:.0f}"
        for q in (0.5, 0.9, 0.95, 0.99)
    )
    print(f"steps  {marks}")
    print(f"score  mean={statistics.mean(scores):.2f}")
    print(f"merges mean={statistics.mean(merges):.1f}")
    print(f"max_type mean={statistics.mean(max_types):.2f}  best={max(max_types):.0f}")
    print(f"corner_wm episodes={sum(1 for r in rows if r['corner_wm'] >= 1)}")
    print(f"win episodes={sum(1 for r in rows if r['win'] >= 1)}")


if __name__ == "__main__":
    main()
