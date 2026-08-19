"""A screen before the A/B: does a weight change move moves 'outside the tie band'?

The inside of the tie band (candidates within `--eps` of the best) was confirmed indifferent at n=133
(NOTES 'Settled: the tie band really is indifferent'). Moves changing inside the band do not move score.
So 'the fraction of moves that change' is not a screen; look at **the fraction that escapes the band**.

Its predictive power was verified on two known failures. `drop_ideal` changes 50.2% of moves but
escapes the band only 6.3%, bumpiness x4 17.5% against 7.0%, and neither
moved score in the n=133 A/B.

The physics runs only one pass and keeps the per-term breakdown of each candidate; the weight sweep is done
analytically from that (reruns take seconds thanks to the `--positions` cache).

Usage:
  python scripts/band_escape.py
  python scripts/band_escape.py --seeds 8 --steps 240 --eps 0.5
  python scripts/band_escape.py --positions artifacts/band_positions.pkl
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT
from src import penalties as pen
from src import policy as pol
from src.observe import Observation, clamp_drop_x
from src.policy import choose_x
from src.reward import merge_score
from src.sim.sim_env import SimEnv
from src.sim.sim_physics import landed_xy, simulate_drop_held
from src.vision.classify import fruit_radius

# The band is measured late in the game. Early boards have few fruits, and neither ladders nor packing have formed.
DEFAULT_SKIP = 60
# Multipliers swept. 1.0 is the identity so it is not included.
MULTIPLIERS = (0.5, 1.5, 2.0)
# Terms that can be swept by a multiplier (the weighted contribution is scaled as is).
SWEEP_KEYS = ("bury", "excess_same", "size_order", "big_layout", "foreign_aim", "variance")


def _components(
    before: list, drop_type: int, x: float, held_r: float, next_type: int | None
) -> tuple[dict[str, float], float]:
    """(term name -> contribution to eval, eval) for one candidate.

    The same breakdown as `policy._evaluate_drop`. That the sum matches `_held_eval`
    is checked by the caller per position (aggregating while it is off breaks the definition of the band).
    """
    sign = pol._order_sign(before)
    after, _merges, merge_types, held_merged = simulate_drop_held(before, drop_type, x)
    land_x, _land_y = landed_xy(before, after, drop_type, x, held_r, held_merged)

    crown = pen._top_crown(after)
    variance = pen._height_variance(after)
    danger = (pen.DANGER_Y - crown) * pen.DANGER_CROWN_WEIGHT if crown < pen.DANGER_Y else 0.0
    if crown < pen.DANGER_Y:
        variance *= pen.VARIANCE_DANGER_SCALE

    parts = {
        "score": merge_score(merge_types),
        "danger": -danger,
        "bury": -pen.BURY_WEIGHT * pen._bury_penalty(after),
        "excess_same": -pen._excess_same_penalty(after),
        "size_order": 0.0 if held_merged else -pen._size_order_penalty(after, sign),
        "big_layout": -pen._big_layout_penalty(after, sign),
        "variance": -pen.VARIANCE_WEIGHT * variance,
        "foreign_aim": -pen.foreign_aim_penalty(before, x, drop_type, held_r),
        "valley_grow": 0.0,
    }
    if not held_merged:
        if pen.valley_grow_ok(before, land_x, drop_type, next_type):
            parts["valley_grow"] = pen.VALLEY_GROW_BONUS
    return parts, sum(parts.values())


def _collect(seeds: list[int], steps: int, skip: int, stride: int) -> list[Observation]:
    positions: list[Observation] = []
    for seed in seeds:
        env = SimEnv(seed=seed)
        obs = env.reset()
        for step in range(steps):
            if obs.held_type is not None and step >= skip and step % stride == 0:
                positions.append(obs)
            result = env.step(choose_x(obs))
            obs = result.observation
            if result.done:
                break
        print(f"  seed {seed}: positions {len(positions)}", flush=True)
    return positions


def _candidate_table(positions: list[Observation]) -> list[list[tuple[float, dict[str, float]]]]:
    """(eval, term breakdown) of every candidate in each position. The only place the physics runs."""
    table = []
    for i, obs in enumerate(positions):
        assert obs.held_type is not None
        before = list(obs.fruits)
        held_r = fruit_radius(obs.held_type)
        rows = []
        first_x: float | None = None
        for x in pol._candidates(before, obs.held_type, held_r, extra_type=obs.next_type):
            x = clamp_drop_x(x, obs.held_type)
            if first_x is None:
                first_x = x
            parts, total = _components(before, obs.held_type, x, held_r, obs.next_type)
            rows.append((total, parts))
        # Check per position, on one candidate, that the breakdown matches the real eval.
        # Aggregating while it is off breaks the definition of the band itself, so do not proceed silently.
        if first_x is not None:
            _after, ref = pol._held_eval(obs, first_x, held_r)
            if abs(rows[0][0] - ref) > 1e-6:
                raise SystemExit(f"breakdown does not match eval: {rows[0][0]} != {ref}")
        table.append(rows)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(positions)}", flush=True)
    return table


def _escape(table, key: str, mult: float, eps: float) -> tuple[int, int]:
    """(positions where the move changed, positions that escaped the band)."""
    changed = escaped = 0
    for rows in table:
        evs = [total for total, _p in rows]
        best = max(evs)
        band = {i for i, e in enumerate(evs) if e >= best - eps}
        base = max(range(len(evs)), key=lambda i: evs[i])
        shifted = [total + parts[key] * (mult - 1.0) for total, parts in rows]
        pick = max(range(len(shifted)), key=lambda i: shifted[i])
        if pick != base:
            changed += 1
        if pick not in band:
            escaped += 1
    return changed, escaped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.1, help="width of the tie band")
    parser.add_argument(
        "--positions",
        type=Path,
        default=ROOT / "artifacts" / "band_positions.pkl",
        help="cache of positions and candidate tables. Reused if present.",
    )
    args = parser.parse_args()

    if args.positions.exists():
        table = pickle.loads(args.positions.read_bytes())
        print(f"reusing cache {args.positions}")
    else:
        seeds = [args.seed + i for i in range(args.seeds)]
        positions = _collect(seeds, args.steps, args.skip, args.stride)
        table = _candidate_table(positions)
        args.positions.parent.mkdir(parents=True, exist_ok=True)
        args.positions.write_bytes(pickle.dumps(table))
        print(f"saved cache: {args.positions}")

    n = len(table)
    sizes = [
        sum(1 for total, _p in rows if total >= max(t for t, _q in rows) - args.eps)
        for rows in table
    ]
    print(f"\n{n} positions   band (eps={args.eps}) candidate count median {statistics.median(sizes):.0f}\n")
    print("term          mult    moves change        escapes the band")
    for key in SWEEP_KEYS:
        for mult in MULTIPLIERS:
            changed, escaped = _escape(table, key, mult, args.eps)
            print(
                f"  {key:<12}x{mult:<4.1f}{changed:>5}/{n} ({changed / n * 100:5.1f}%)"
                f"   {escaped:>5}/{n} ({escaped / n * 100:5.1f}%)"
            )


if __name__ == "__main__":
    main()
