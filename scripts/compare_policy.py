"""Pit two bootstrap variants against each other on the same seeds and report the difference by phase.

A tool for localizing 'where it got better/worse' after a change.
The default compares placement after the floor fills (SUIKA_PACKED) ON/OFF. The same seed sequence goes through both,
reporting not just means but per-seed wins and losses and metrics split into early and late game.

Usage:
  python scripts/compare_policy.py
  python scripts/compare_policy.py --episodes 60 --max-steps 120 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT

from src.parallel import default_workers
from src.reward import watermelon_count

# Moves considered early game. How the board breaks up to here is looked at separately from later.
EARLY_STEPS = 30
# Merges per move counted as a cascade firing.
CASCADE_MERGES = 3
# Where run results go. The key includes the hash of policy.py, so leaving them does not go stale.
CACHE_DIR = ROOT / "artifacts" / "compare_cache"


def _episode(seed: int, max_steps: int, variant: bool) -> dict[str, float]:
    """Run one episode and return metrics. Runs on the ProcessPool worker side."""
    from src import policy
    from src.policy import choose_x
    from src.sim_env import SimEnv

    policy.set_packed_rule_enabled(variant)

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
        result = env.step(choose_x(obs))
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


def _policy_digest() -> str:
    """A hash of the contents of policy.py. The cache is invalidated when it changes."""
    src = (ROOT / "src" / "policy.py").read_bytes()
    return hashlib.sha256(src).hexdigest()[:16]


def _cache_path(seeds: list[int], max_steps: int, variant: bool) -> Path:
    key = f"{_policy_digest()}-{seeds[0]}-{len(seeds)}-{max_steps}-{int(variant)}"
    return CACHE_DIR / f"{key}.json"


def _run(
    seeds: list[int],
    max_steps: int,
    variant: bool,
    workers: int,
    *,
    label: str,
) -> list[dict[str, float]]:
    """Run one variant. With the same seeds the result is determined, so it is cached.

    The OFF side is the same across variants. Recomputing it every time doubles the A/B run time.
    The cache key includes the hash of policy.py, so touching the policy
    invalidates it automatically. It also prevents comparing against a stale baseline.
    """
    path = _cache_path(seeds, max_steps, variant)
    if path.is_file():
        print(f"  {label}: using cache ({path.name})", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))

    started = time.monotonic()
    print(f"  {label}: running {len(seeds)} episodes...", flush=True)
    if workers <= 1:
        rows = [_episode(s, max_steps, variant) for s in seeds]
    else:
        # The child reads the policy from the environment at spawn. Also overwritten with the setter.
        os.environ["SUIKA_PACKED"] = "1" if variant else "0"
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

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [r[key] for r in rows if r[key] == r[key]]
    return statistics.mean(values) if values else float("nan")


def _line(label: str, a: float, b: float, *, digits: int = 2) -> str:
    delta = b - a
    pct = f"{delta / a * 100:+6.1f}%" if a else "     -"
    return f"  {label:<14} {a:9.{digits}f} -> {b:9.{digits}f}  ({delta:+.{digits}f} {pct})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else default_workers()

    seeds = [args.seed + i for i in range(args.episodes)]
    base = _run(seeds, args.max_steps, False, workers, label="A (OFF)")
    new = _run(seeds, args.max_steps, True, workers, label="B (ON) ")

    print(
        f"\nepisodes={args.episodes} seed={args.seed} "
        f"max_steps={args.max_steps} workers={workers}"
    )
    print("  A = placement after the floor fills OFF (current)   B = ON")

    # If everything is truncated, neither headroom nor death rate is measured.
    # It becomes a measurement looking only at setup cost and not the return, so warn first.
    capped = sum(
        1 for row in base + new if row["steps"] >= args.max_steps
    )
    if capped == len(base) + len(new):
        print(
            f"\n  ** all {capped} episodes truncated at max_steps={args.max_steps}. **\n"
            "  ** Not one natural end, so neither survival time nor stage reached is measured. **\n"
            "  ** Increase --max-steps and measure again. Do not trust the numbers below. **"
        )
    for label, key, digits in (
        ("score", "score", 2),
        ("early_score", "early_score", 2),
        ("steps", "steps", 1),
        ("merges", "merges", 1),
        ("cascades", "cascades", 2),
        ("max_type", "max_type", 2),
        ("early_crown", "early_crown", 1),
        ("dead", "dead", 3),
        ("dead_early", "dead_early", 3),
    ):
        print(_line(label, _mean(base, key), _mean(new, key), digits=digits))

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
