"""A tool that rereads compare_policy --out dumps to choose a proxy metric.

score is noisy, and speaking to ±100 points needs n≈100. One A/B takes hours,
which is the bottleneck of improving the policy. So look for 'a metric that captures the same change
with fewer episodes than score'.

For each metric, from the paired differences on the same seeds:

- n_detect: episodes needed to move the observed difference away from 0 at the 95% CI.
  **The smaller this is, the sooner the same change is detected = better suited as a proxy.**
- sensitivity ratio: |t| / |t of score|. Above 1 means sharper than score.
- pairing gain: how much pairing the seeds shrank the standard error of the difference. On independent
  seeds the SD of the difference is sqrt(2) times the per-game variation, so that is the reference.
  0 means pairing did nothing, 0.5 means the paired comparison halves the SE.
- r: correlation with score. A metric near 0 may be fast to measure but measures something else.

**Beware of selection bias.** Picking the metric that looked best on the same data picks
something that just drew a lucky hand. Before adopting a metric that ranks high here, confirm it also ranks high
on a second dump taken with a different change and a different seed sequence.
Listing several dumps as arguments prints rows for every dump per metric.

Usage:
  python scripts/analyze_ab.py artifacts/packed_ab_n100.json
  python scripts/analyze_ab.py artifacts/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import correlation, detect_n, paired_stats, pairing_gain

# seed is not a metric.
SKIP = {"seed"}


def _fmt(value: float, digits: int = 2, width: int = 8) -> str:
    return f"{'-':>{width}}" if value != value else f"{value:>{width}.{digits}f}"


def _report(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    a_rows: list[dict[str, float]] = data["a"]
    b_rows: list[dict[str, float]] = data["b"]
    keys = [k for k in a_rows[0] if k not in SKIP]

    scores_a = [r["score"] for r in a_rows]
    scores_b = [r["score"] for r in b_rows]
    t_score = abs(paired_stats(scores_a, scores_b).t)

    print(
        f"\n{path.name}: {data.get('variant', '?')} "
        f"episodes={data.get('episodes', len(a_rows))} "
        f"max_steps={data.get('max_steps', '?')} seed={data.get('seed', '?')}"
    )
    capped = sum(
        1 for r in a_rows + b_rows if r["steps"] >= float(data.get("max_steps", 0))
    )
    if capped:
        print(f"  note: {capped}/{len(a_rows) + len(b_rows)} truncated at max_steps")
    print(
        "  metric         delta   sd_diff  n_detect  sens.ratio  pair.gain   r(score)"
    )
    rows = []
    for key in keys:
        stats = paired_stats([r[key] for r in a_rows], [r[key] for r in b_rows])
        gain = pairing_gain(stats.sd_diff, stats.sd_pooled)
        rows.append(
            (
                detect_n(stats.delta, stats.sd_diff),
                key,
                stats,
                gain,
                correlation(
                    [r[key] for r in a_rows] + [r[key] for r in b_rows],
                    scores_a + scores_b,
                ),
            )
        )
    # Ascending n_detect = the order in which metrics settle fastest. NaN (no difference) goes last.
    rows.sort(key=lambda r: (r[0] != r[0], r[0]))
    for detect, key, stats, gain, r in rows:
        sens = abs(stats.t) / t_score if t_score else float("nan")
        print(
            f"  {key:<12}{_fmt(stats.delta, 2, 9)} {_fmt(stats.sd_diff, 2)}"
            f" {_fmt(detect, 0, 9)} {_fmt(sens, 2)} {_fmt(gain, 2, 11)}"
            f" {_fmt(r, 2, 10)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.dumps:
        _report(path)
    if len(args.dumps) > 1:
        print(
            "\n  Adopt as a proxy only a metric that ranks high consistently across several dumps."
        )


if __name__ == "__main__":
    main()
