"""Pit two bootstrap variants against each other and report the difference by phase.

A tool for localizing 'where it got better/worse' after a change.
The same seed sequence is run through both A and B, reporting not just means but per-seed wins and losses,
and metrics split into early and late game.

**Plug the change you want to compare into `_apply_variant`.** Left empty, A and B
are the same policy and a warning that every seed tied appears.

Per metric, the paired t value and 95% CI of the difference on the same seeds are reported (`src/stats.py`).
This prevents reading 'it got better' from a rise or fall in the mean; a row whose CI crosses 0
says nothing at that n. The n@5% column is the number of episodes needed to speak to that metric
at 5% precision; the smaller, the fewer runs needed to read a change.

Omitting --seed starts from a random seed every time. Reusing fixed seeds over and over
makes it easy to misread a collapse that happened by chance on that seed set as 'reproduced'.

Usage:
  python scripts/compare_policy.py
  python scripts/compare_policy.py --episodes 60 --max-steps 300 --workers 8
  python scripts/compare_policy.py --episodes 100 --max-steps 400 --out artifacts/ab.json
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.util.parallel import default_workers
from src.reward import watermelon_count
from src.util.stats import correlation, paired_stats

# Moves considered early game. How the board breaks up to here is looked at separately from later.
EARLY_STEPS = 30
# Merges per move counted as a cascade firing.
CASCADE_MERGES = 3


def _apply_variant(enabled: bool) -> None:
    """Switch the change to compare here (side B is enabled=True).

    Called in each worker process before running episodes. Leaving toggles for permanent rules
    adds dead branches, so empty the body once the experiment is over.
    Left empty, A and B are the same policy and a warning that every seed tied appears.

    Weights are swapped by rewriting module attributes. Penalty constants are in src.penalties,
    search coarseness and the lookahead discount in src.policy.

    Example:
        from src import penalties
        penalties.BURY_WEIGHT = 30.0 if enabled else 20.0
    """


def _episode(
    seed: int,
    max_steps: int,
    variant: bool,
    pool: Executor | None = None,
) -> dict[str, float]:
    """Run one episode and return metrics. Runs on the ProcessPool worker side.

    pool is passed from _run only when 'there is nothing to parallelize per episode (workers<=1)'.
    The side that spreads episodes themselves over a ProcessPool leaves it None to avoid double parallelism.
    """
    from src.policy import choose_x
    from src.sim.sim_env import SimEnv

    _apply_variant(variant)

    env = SimEnv(seed=seed)
    obs = env.reset()
    score = 0.0
    early_score = 0.0
    merges = 0
    cascades = 0
    steps = 0
    max_type = -1
    max_wm = 0
    early_crowns: list[float] = []
    info = "ok"
    for _ in range(max_steps):
        result = env.step(choose_x(obs, pool=pool))
        obs = result.observation
        score += result.score
        merges += result.merges
        steps += 1
        info = result.info
        if result.merges >= CASCADE_MERGES:
            cascades += 1
        if steps <= EARLY_STEPS:
            early_score += result.score
            if obs.fruits:
                early_crowns.append(min(f.y - f.radius for f in obs.fruits))
        if obs.fruits:
            max_type = max(max_type, max(f.type for f in obs.fruits))
        max_wm = max(max_wm, watermelon_count(obs))
        if result.done:
            break
    return {
        "seed": float(seed),
        "steps": float(steps),
        "score": score,
        "early_score": early_score,
        "merges": float(merges),
        "cascades": float(cascades),
        "max_type": float(max_type),
        "max_wm": float(max_wm),
        # y points down. Smaller means a taller pile = dangerous.
        "early_crown": min(early_crowns) if early_crowns else float("nan"),
        # Whether it died before reaching the cap. A direct metric of early collapse.
        "dead_early": 1.0 if (info == "dead" and steps <= EARLY_STEPS) else 0.0,
        "dead": 1.0 if info == "dead" else 0.0,
        "win": 1.0 if info == "win" else 0.0,
    }


def _run(
    seeds: list[int],
    max_steps: int,
    variant: bool,
    workers: int,
    *,
    label: str,
) -> list[dict[str, float]]:
    started = time.monotonic()
    print(f"  {label}: running {len(seeds)} episodes...", flush=True)
    if workers <= 1:
        # With nothing to parallelize per episode, parallelize choose_x candidate evaluation instead.
        with ProcessPoolExecutor() as move_pool:
            rows = [_episode(s, max_steps, variant, pool=move_pool) for s in seeds]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_episode, s, max_steps, variant) for s in seeds
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                print(
                    f"    {done}/{len(seeds)}  ({time.monotonic() - started:.0f}s)",
                    end="\r",
                    flush=True,
                )
        rows.sort(key=lambda r: r["seed"])
    print(f"  {label}: done {time.monotonic() - started:.0f}s" + " " * 20, flush=True)
    return rows


def _column(rows: list[dict[str, float]], key: str) -> list[float]:
    return [r[key] for r in rows]


def _fmt(value: float, digits: int) -> str:
    return "-" if value != value else f"{value:.{digits}f}"


def _line(
    label: str,
    base: list[dict[str, float]],
    new: list[dict[str, float]],
    *,
    digits: int = 2,
) -> str:
    """The row for one metric. Not just means but the paired t, 95% CI and required n.

    n@5% is 'the number of episodes needed to speak to that metric at 5% precision'. It is relative
    so that metrics are compared on the same yardstick. The smaller it is,
    the fewer episodes needed to read a change = better suited as a proxy.
    r is the correlation of the metric in that row with score, a guide to whether it means anything as a proxy.
    """
    stats = paired_stats(_column(base, label), _column(new, label))
    pct = f"{stats.delta / stats.mean_a * 100:+6.1f}%" if stats.mean_a else "     -"
    ci = (
        f"[{stats.ci_lo:+8.1f},{stats.ci_hi:+8.1f}]"
        if stats.ci_lo == stats.ci_lo
        else " " * 19
    )
    n5 = stats.required_n(abs(stats.mean_a) * 0.05) if stats.mean_a else float("nan")
    r = correlation(
        _column(base, label) + _column(new, label),
        _column(base, "score") + _column(new, "score"),
    )
    mark = " *" if stats.significant else "  "
    return (
        f"  {label:<12}{_fmt(stats.mean_a, digits):>9} ->{_fmt(stats.mean_b, digits):>9}"
        f" ({pct}) t={_fmt(stats.t, 2):>6} {ci}{mark}"
        f" n@5%={_fmt(n5, 0):>7} r={_fmt(r, 2):>5}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random every time when omitted. Avoid reusing fixed seeds.",
    )
    # The old default of 100 was half a natural game (median 210 moves), and all 200 measured runs were truncated.
    # The default measured only the cost of setting up and not the return, so it is 400.
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="save per-seed raw data as JSON (so metrics can be chosen again later).",
    )
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else default_workers()
    seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000)

    seeds = [seed + i for i in range(args.episodes)]
    base = _run(seeds, args.max_steps, False, workers, label="A (base)")
    new = _run(seeds, args.max_steps, True, workers, label="B (new) ")

    # Save before aggregating. Raw data from a long run must not be lost to a trivial bug on the aggregation side.
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "episodes": args.episodes,
                    "max_steps": args.max_steps,
                    "variant": _apply_variant.__doc__.splitlines()[0],
                    "a": base,
                    "b": new,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"  saved raw data: {args.out}")

    print(
        f"\nepisodes={args.episodes} seed={seed} "
        f"max_steps={args.max_steps} workers={workers}"
    )
    print("  A = _apply_variant(False)   B = _apply_variant(True)")

    # Truncation caps the runs that went long, so the better the change the more it is underestimated.
    # Warning only on 'all truncated' would let partial truncation pass without warning.
    total = len(base) + len(new)
    capped = sum(1 for row in base + new if row["steps"] >= args.max_steps)
    if capped == total:
        print(
            f"\n  ** all {capped} episodes truncated at max_steps={args.max_steps}. **\n"
            "  ** Not one natural end, so neither survival time nor stage reached is measured. **\n"
            "  ** Increase --max-steps and measure again. Do not trust the numbers below. **"
        )
    elif capped:
        print(
            f"\n  ** {capped}/{total} episodes truncated at max_steps={args.max_steps}"
            f" ({capped / total * 100:.0f}%). **\n"
            "  ** The truncated ones are games that went long, so the difference is compressed by that much. **"
        )
    print()
    for key, digits in (
        ("score", 2),
        ("early_score", 2),
        ("steps", 1),
        ("merges", 1),
        ("cascades", 2),
        ("max_type", 2),
        ("early_crown", 1),
        ("dead", 3),
        ("dead_early", 3),
    ):
        print(_line(key, base, new, digits=digits))
    print(
        "  (* = 95% CI does not cross 0 / n@5%: episodes needed to speak to a 5% difference"
        " / r: correlation with score)"
    )

    # The real verdict is the paired score difference. Only here is the conclusion written out.
    score_stats = paired_stats(_column(base, "score"), _column(new, "score"))
    print(
        f"\n  score paired difference: {score_stats.delta:+.1f} "
        f"(SD of the difference={_fmt(score_stats.sd_diff, 1)}, SE={_fmt(score_stats.se, 1)})"
    )
    if score_stats.sd_diff == 0.0:
        print("  score is identical on every pair. No difference, no test needed.")
    elif score_stats.significant:
        print("  95% CI does not cross 0. Significant at this n.")
    else:
        need = score_stats.required_n(100.0)
        print(
            "  95% CI crosses 0 = no significant difference at this n."
            f" Speaking to +/-100 points needs about n={_fmt(need, 0)}."
        )

    # Per-seed head-to-head. Even if the mean does not move, split wins and losses mean something different.
    wins = sum(1 for a, b in zip(base, new) if b["score"] > a["score"])
    losses = sum(1 for a, b in zip(base, new) if b["score"] < a["score"])
    ties = args.episodes - wins - losses
    print(f"\n  seed head-to-head (B's view)  win={wins}  loss={losses}  tie={ties}")
    if ties == args.episodes:
        print("  ** identical on every seed. The change does not fire under these conditions **")

    worst = sorted(zip(base, new), key=lambda p: p[1]["score"] - p[0]["score"])[:5]
    print("\n  seeds that worsened most:")
    for a, b in worst:
        if b["score"] >= a["score"]:
            break
        print(
            f"    seed={int(a['seed']):4d} score {a['score']:7.0f} -> {b['score']:7.0f}"
            f"  steps {a['steps']:3.0f} -> {b['steps']:3.0f}"
            f"  max_type {a['max_type']:.0f} -> {b['max_type']:.0f}"
        )


if __name__ == "__main__":
    main()
