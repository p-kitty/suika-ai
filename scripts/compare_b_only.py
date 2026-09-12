"""Run only side B and pair it with a saved baseline (side A).

Side A is the same policy across variants, so rerunning it for every A/B is pure waste.
Run only B on the same seed sequence and pair it with side A from the `--baseline` JSON.
The statistics stay the same paired test as `compare_policy`, at half the compute.

Pairing itself helps little (measured r=+0.11 to +0.18, an SE gain of 6-10%).
Changing one move makes the board diverge, so even on the same seed A and B are nearly uncorrelated.
Pairing is still kept to match the draw luck of the seed set on both sides.

**A baseline becomes invalid when the policy changes.** A warning is shown when the JSON commit and the current HEAD
differ, so do not ignore it and use it anyway.

Plug the change into `_apply_variant` in `compare_policy.py`. Only side B calls it.

Usage:
  python scripts/compare_policy.py --episodes 50 --out artifacts/base.json
  python scripts/compare_b_only.py --baseline artifacts/base.json --episodes 50
  python scripts/compare_b_only.py --baseline artifacts/base.json --offset 50 --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_policy import _apply_variant, _episode, _head, _line

METRICS = (
    ("score", 2),
    ("early_score", 2),
    ("steps", 1),
    ("merges", 1),
    ("cascades", 2),
    ("max_type", 2),
    ("early_crown", 1),
    ("dead", 3),
    ("dead_early", 3),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="number of baseline episodes to skip from the start (rerun the same variant on another seed band)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.baseline.read_text(encoding="utf-8"))
    base_all = data["a"]
    max_steps = data["max_steps"]
    if args.offset + args.episodes > len(base_all):
        raise SystemExit(
            f"the baseline has only {len(base_all)} episodes: "
            "lower --episodes or --offset"
        )
    base = sorted(base_all, key=lambda r: r["seed"])[
        args.offset : args.offset + args.episodes
    ]
    seeds = [int(r["seed"]) for r in base]

    stamp = data.get("baseline_commit")
    if stamp and stamp != _head():
        print(f"  warning: baseline commit {stamp[:8]} differs from the current HEAD", flush=True)

    started = time.monotonic()
    print(f"  B (new): running {len(seeds)} episodes... (side A reused)", flush=True)
    rows: list[dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_episode, s, max_steps, True) for s in seeds]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"    {done}/{len(seeds)}  ({time.monotonic() - started:.0f}s)",
                  end="\r", flush=True)
    rows.sort(key=lambda r: r["seed"])
    print(f"  B (new): done {time.monotonic() - started:.0f}s" + " " * 20, flush=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "baseline": str(args.baseline),
                    "episodes": args.episodes,
                    "offset": args.offset,
                    "max_steps": max_steps,
                    "variant": (_apply_variant.__doc__ or "").strip().splitlines()[:1],
                    "a": base,
                    "b": rows,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"  saved raw data: {args.out}")

    print(f"\nepisodes={args.episodes} max_steps={max_steps} "
          f"workers={args.workers} baseline={args.baseline.name}")
    print(
        f"  A = saved baseline (of {len(base_all)} episodes, "
        f"{args.episodes} starting at episode {args.offset})"
    )
    for key, digits in METRICS:
        print(_line(key, base, rows, digits=digits))
    print("  (* = 95% CI does not cross 0)")

    from src.util.stats import paired_stats

    stats = paired_stats([r["score"] for r in base], [r["score"] for r in rows])
    print(f"\n  score paired difference: {stats.delta:+.1f} "
          f"(SD of the difference={stats.sd_diff:.1f}, SE={stats.se:.1f})")
    wins = sum(1 for x, y in zip(base, rows) if y["score"] > x["score"])
    losses = sum(1 for x, y in zip(base, rows) if y["score"] < x["score"])
    print(f"  seed head-to-head (B's view)  win={wins}  loss={losses}  "
          f"tie={len(base) - wins - losses}")


if __name__ == "__main__":
    main()
