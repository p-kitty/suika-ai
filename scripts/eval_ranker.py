"""Play the sim with the BC ranker in place of the teacher's eval, and report score.

NOTES 'Ranking candidates by their own post-drop features breaks the BC ceiling' measured agreement
offline. The RL gate has a second half -- "the student's score close to bootstrap" -- and that needs
the ranker to actually pick moves. Here every candidate's post-drop board is scored by the fitted
weights and the best is played.

The ranker is not a speedup: its features are post-drop, so it simulates every candidate exactly as
the teacher does. This measures quality, not cost.

Usage:
  python scripts/eval_ranker.py --episodes 8 --workers 8
  python scripts/eval_ranker.py --episodes 8 --weights artifacts/ranker_0916.npz
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT
from scripts._episodes import play_episode
from src import policy as pol
from src.observe import Observation
from src.training.features import board_features
from src.util.parallel import resolve_workers

DEFAULT_WEIGHTS = ROOT / "artifacts" / "ranker_0916.npz"
_W: dict[str, np.ndarray] = {}


def _load(path: Path) -> None:
    if "w" not in _W:
        d = np.load(path)
        _W["w"], _W["mean"], _W["sd"] = d["w"], d["mean"], d["sd"]


def _choose(obs: Observation, weights: Path, scorer: str) -> float:
    """Best candidate by `scorer`, with no next lookahead either way.

    scorer "oneply" takes the teacher's own first-ply eval, which is what `rank_candidates` already
    returns sorted. It is the control that separates "the ranker imitates the first ply badly" from
    "the missing second ply is the whole gap".
    """
    ranked = pol.rank_candidates(obs)
    if not ranked:
        return pol.choose_x(obs)
    if scorer == "oneply":
        return ranked[0][1]
    _load(weights)
    sign = pol._order_sign(list(obs.fruits))
    best_x, best_s = ranked[0][1], -float("inf")
    for _held_eval, x, after, _score in ranked:
        z = (board_features(after, sign=sign) - _W["mean"]) / _W["sd"]
        s = float(z @ _W["w"])
        if s > best_s:
            best_s, best_x = s, x
    return best_x


def _episode(seed: int, max_steps: int, weights: Path, scorer: str) -> dict[str, float]:
    return play_episode(seed, lambda obs: _choose(obs, weights, scorer), max_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=930000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--scorer", choices=("ranker", "oneply"), default="ranker",
                        help="oneply plays the teacher's first-ply eval, the control for the missing lookahead")
    args = parser.parse_args()
    workers = resolve_workers(args.workers)

    seeds = [args.seed + i for i in range(args.episodes)]
    started = time.monotonic()
    rows: list[dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_episode, s, args.max_steps, args.weights, args.scorer) for s in seeds]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"  {done}/{len(seeds)} ({time.monotonic() - started:.0f}s)", end="\r", flush=True)
    print(f"  done {time.monotonic() - started:.0f}s" + " " * 20)

    for key in ("score", "steps", "merges", "max_type"):
        vals = [r[key] for r in rows]
        print(f"  {key:<10} mean {statistics.mean(vals):8.2f}   median {statistics.median(vals):8.2f}")
    truncated = sum(1 for r in rows if r["steps"] >= args.max_steps)
    print(f"  truncated {truncated}/{len(rows)}")


if __name__ == "__main__":
    main()
