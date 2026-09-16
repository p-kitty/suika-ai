"""Collect training data for the value function. Trajectories of the teacher (`choose_x`) + realized returns.

What `train_sim.py bc` collects is 'observation -> teacher action', which was measured to
plateau at match 30% (NOTES 'Investigated: BC does not reach 60-70% match'). This one
collects '**post-drop board** -> **points actually scored afterwards**'. The teacher's eval is
not regressed because the inside of the tie band was confirmed truly indifferent at n=133
(NOTES 'Settled: the tie band really is indifferent'), so approximating eval hits the same ceiling.

The candidate table is also kept every `--candidate-stride` moves. It is for verifying rankings (whether the learned V
puts an order into the teacher's tie band), not a training label.

`--workers` defaults to logical cores/2. Raising it makes the real game stutter on the same machine
(→AGENTS.md 'When running something that fills the CPU').

Usage:
  python scripts/collect_value.py --episodes 4 --max-steps 60
  python scripts/collect_value.py --episodes 100 --out artifacts/value_100.npz
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT

from src.training.collect import (
    CANDIDATE_STRIDE,
    collect_value_episodes,
    save_value_dataset,
)
from src.training.features import FEATURE_NAMES
from src.util.parallel import resolve_workers

DEFAULT_OUT = ROOT / "artifacts" / "value_dataset.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    # 400 to match the other scripts. At 300, 5-10% are truncated, and the returns
    # come out missing (the truncation item in NOTES 'How to measure').
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None,
                        help="parallelism (logical cores/2 when omitted, 1 for serial)")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--candidate-stride",
        type=int,
        default=CANDIDATE_STRIDE,
        help=f"interval for keeping candidate tables (default {CANDIDATE_STRIDE}, 0 keeps none)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    workers = resolve_workers(args.workers)
    print(
        f"=== collect value ({args.episodes} ep, max_steps={args.max_steps}, "
        f"workers={workers}, stride={args.candidate_stride}) ===",
        flush=True,
    )
    started = time.time()
    data = collect_value_episodes(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        workers=workers,
        log_every=args.log_every,
        candidate_stride=args.candidate_stride,
    )
    elapsed = time.time() - started

    n = len(data.feats)
    if n == 0:
        raise SystemExit("not a single move was collected")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_value_dataset(data, args.out)

    # Per-episode summary. Truncation is needed to decide what the learner drops.
    ep_ids = sorted(set(data.episodes.tolist()))
    totals = [float(data.rewards[data.episodes == e].sum()) for e in ep_ids]
    lengths = [int((data.episodes == e).sum()) for e in ep_ids]
    cut = sum(1 for e in ep_ids if bool(data.truncated[data.episodes == e][0]))
    corner = sum(1 for e in ep_ids if bool(data.cornered[data.episodes == e][0]))

    print(f"\nsaved: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"moves {n}   candidate rows {len(data.cand_row)}   {elapsed / 60:.1f} min")
    print(
        f"score mean {statistics.fmean(totals):.1f}   "
        f"moves mean {statistics.fmean(lengths):.1f}"
    )
    print(f"truncated {cut}/{len(ep_ids)}   corner watermelons {corner}/{len(ep_ids)}")
    print(f"return mean {data.returns.mean():.1f}  max {data.returns.max():.1f}")
    print(f"\n{len(FEATURE_NAMES)} features  (mean / standard deviation)")
    for i, name in enumerate(FEATURE_NAMES):
        col = data.feats[:, i]
        print(f"  {name:<18}{col.mean():9.3f} {col.std():9.3f}")


if __name__ == "__main__":
    main()
