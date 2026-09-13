"""A screen before the A/B: does adding a board score to the two-ply value move moves outside 'the two-ply band'?

A different role from `band_escape.py`. That one cuts the band on the **first-ply eval**, but the current policy
ranks `HELD_TOP` candidates by the two-ply value (held eval + `NEXT_DISCOUNT` × best next),
so looking at the first-ply band does not show the current policy's band
(→NOTES 'Measured and dropped: third-ply expectation'). Here the same 8 are
reordered by Q = two-ply value + λ·V(post-drop board), reporting **the fraction escaping the band** per λ.

Escaping the band is only a necessary condition (the learned V that escaped 18.9% was also null at n=150).
So two things are reported before the λ sweep:

- **Are the boards in the band really different boards?** Candidates differing only by physics jitter (under 1px)
  cannot be given a meaningful order by any board score
- **Does each feature split inside the band?** A good fit guarantees nothing about traction between candidates
  (the terms that supported the learned V's fit did not move even once inside the band)

The physics runs one pass, and the cache keeps **the post-drop boards themselves**. Swapping features or models
needs no replay (seconds with `--positions`).

Usage:
  python scripts/value_escape.py --model artifacts/value_h100.npz
  python scripts/value_escape.py --model artifacts/value_h100.npz --eps 0.5
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT
from src import policy as pol
from src.policy import choose_x, rank_candidates
from src.reward import is_lost
from src.sim.sim_env import SimEnv
from src.training.features import FEATURE_NAMES, board_features
from src.training.value import LinearValue, load as load_value
from src.vision.state import Fruit

# λ values swept. V's candidate range is orders of magnitude larger than the eval band width (0.1),
# so unless swept at close to 10x steps it jumps between 'no effect' and 'V takes over'.
LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
# Shift between matched fruits considered the same board. The median difference of the biggest fruit between candidates is 0.12px,
# which was physics jitter, not a choice (NOTES 'Continuous corner and height terms').
# Set one order of magnitude above that, folding candidates that differ only by jitter into the same board.
SAME_BOARD_PX = 2.0
# Move number considered late game. Matches the `--skip` default of `band_escape.py`.
LATE_STEP = 60
# Difference for which a feature value counts as split. Features are normalized to around 0-1, so
# the floor is well below 1px / screen width (about 1/400).
FEATURE_SPLIT = 1e-4


@dataclass
class Position:
    """The candidate table of one position. **It keeps the boards themselves**, so swapping features needs no replay."""

    seed: int
    step: int
    sign: int  # direction of the board **before** the drop (same basis as the collection side)
    values: np.ndarray  # (K,) two-ply values
    boards: list[list[Fruit]]  # (K,) post-drop boards
    teacher: int  # index chosen by the unmodified policy


def _position(obs, seed: int, step: int) -> Position | None:
    """The candidate table of one position.

    The candidate order and the `HELD_TOP` cut must be the same as `choose_x` or the band definition
    breaks, so the result of `rank_candidates` is used as is, the move is decided by `choose_x`
    and the two are checked against each other (the move-selection rules are not copied here).
    """
    assert obs.held_type is not None
    ranked = rank_candidates(obs)
    if not ranked:
        return None
    x = choose_x(obs, ranked=ranked)

    alive = [row for row in ranked if not is_lost(row[2])]
    if alive:
        ranked = alive
    top = ranked[: pol.HELD_TOP]
    if len(top) < 2:
        return None

    boards = [after for _eval, _x, after, _score in top]
    if obs.next_type is None:
        next_scores = [0.0] * len(top)
    else:
        next_scores = pol._best_next_scores(
            boards, obs.next_type, step=pol.NEXT_CANDIDATE_STEP
        )
    values = np.array(
        [
            held_eval + pol.NEXT_DISCOUNT * next_score
            for (held_eval, _x, _after, _score), next_score in zip(top, next_scores)
        ]
    )
    # Aggregating while it is off breaks the definition of the band itself, so do not proceed silently.
    teacher = int(np.argmax(values))
    if top[teacher][1] != x:
        raise SystemExit(f"two-ply argmax differs from choose_x: {top[teacher][1]} != {x}")
    return Position(
        seed=seed,
        step=step,
        sign=pol._order_sign(list(obs.fruits)),
        values=values,
        boards=[list(board) for board in boards],
        teacher=teacher,
    )


def _collect(seeds: list[int], steps: int, skip: int, stride: int) -> list[Position]:
    table: list[Position] = []
    for seed in seeds:
        env = SimEnv(seed=seed)
        obs = env.reset()
        for step in range(steps):
            if obs.held_type is None:
                break
            if step >= skip and step % stride == 0:
                row = _position(obs, seed, step)
                if row is not None:
                    table.append(row)
            result = env.step(choose_x(obs))
            obs = result.observation
            if result.done:
                break
        print(f"  seed {seed}: positions {len(table)}", flush=True)
    return table


def _board_shift(a: list[Fruit], b: list[Fruit]) -> float | None:
    """The difference between two boards. None if the type lists differ (a different merge outcome = a different board).

    If the same, the max shift (px) when same-type fruits are matched nearest first. Greedy matching
    is not optimal, but it is enough to separate jitter (under 1px) from a choice (a dozen px or more).
    """
    if sorted(f.type for f in a) != sorted(f.type for f in b):
        return None
    used: set[int] = set()
    worst = 0.0
    for fa in a:
        best, best_d = -1, float("inf")
        for j, fb in enumerate(b):
            if j in used or fb.type != fa.type:
                continue
            d = float(np.hypot(fa.x - fb.x, fa.y - fb.y))
            if d < best_d:
                best, best_d = j, d
        used.add(best)
        worst = max(worst, best_d)
    return worst


def _same_board(a: list[Fruit], b: list[Fruit]) -> bool:
    shift = _board_shift(a, b)
    return shift is not None and shift <= SAME_BOARD_PX


def _distinct_boards(boards: list[list[Fruit]]) -> tuple[int, list[float | None]]:
    """The number of distinct boards in the band, and the differences between representatives (None = different merge outcome)."""
    reps: list[list[Fruit]] = []
    for board in boards:
        if not any(_same_board(board, rep) for rep in reps):
            reps.append(board)
    shifts = [
        _board_shift(reps[i], reps[j])
        for i in range(len(reps))
        for j in range(i + 1, len(reps))
    ]
    return len(reps), shifts


def _report_band(table: list[Position], feats: list[np.ndarray], eps: float) -> None:
    n = len(table)
    bands = [p.values >= p.values.max() - eps for p in table]
    sizes = [int(band.sum()) for band in bands]
    all_tied = sum(1 for p, size in zip(table, sizes) if size == len(p.values))
    late = sum(1 for p in table if p.step >= LATE_STEP)
    print(
        f"\n{n} positions (late step>={LATE_STEP}: {late})   "
        f"candidates (HELD_TOP) median {statistics.median(len(p.values) for p in table):.0f}"
    )
    print(
        f"two-ply band (eps={eps}) candidate count median {statistics.median(sizes):.0f}"
        f"   all candidates in the band {all_tied}/{n} ({all_tied / n * 100:.1f}%)"
    )

    # --- Are the boards in the band really different boards ---
    multi = [(p, band) for p, band in zip(table, bands) if int(band.sum()) >= 2]
    counts: list[int] = []
    merge_differs = 0
    pair_shifts: list[float] = []
    for p, band in multi:
        count, shifts = _distinct_boards([b for b, keep in zip(p.boards, band) if keep])
        counts.append(count)
        if any(s is None for s in shifts):
            merge_differs += 1
        pair_shifts.extend(s for s in shifts if s is not None)
    m = len(multi)
    print(
        f"\n=== boards in the band ({m} positions with 2+ in the band; within {SAME_BOARD_PX}px is the same board) ==="
    )
    if m:
        one = sum(1 for c in counts if c == 1)
        five = sum(1 for c in counts if c >= 5)
        print(f"  distinct boards median {statistics.median(counts):.0f}")
        print(f"  collapse to one board  {one:>5}/{m} ({one / m * 100:5.1f}%)")
        print(f"  5 or more boards       {five:>5}/{m} ({five / m * 100:5.1f}%)")
        print(f"  merge outcome differs  {merge_differs:>5}/{m} ({merge_differs / m * 100:5.1f}%)")
        if pair_shifts:
            print(
                f"  max shift between same-type boards median {statistics.median(pair_shifts):.1f}px"
                f"   (pairs {len(pair_shifts)})"
            )

    # --- Does each feature split inside the band ---
    print(f"\n=== does each feature split inside the band (difference > {FEATURE_SPLIT:g}) ===")
    print("  feature             positions split     median range inside the band (split positions)")
    for i, name in enumerate(FEATURE_NAMES):
        spans = []
        for f, band in zip(feats, bands):
            if int(band.sum()) < 2:
                continue
            col = f[band, i]
            spans.append(float(col.max() - col.min()))
        split = [s for s in spans if s > FEATURE_SPLIT]
        med = f"{statistics.median(split):.4f}" if split else "-"
        frac = len(split) / max(len(spans), 1) * 100
        print(f"  {name:<18}{len(split):>5}/{len(spans)} ({frac:5.1f}%)   {med:>10}")


def _sweep(
    table: list[Position], feats: list[np.ndarray], model: LinearValue, eps: float
) -> None:
    vs = [model.predict(f) for f in feats]
    v_ranges = [float(v.max() - v.min()) for v in vs]
    n = len(table)
    n_late = sum(1 for p in table if p.step >= LATE_STEP)
    print(f"\nV candidate range median {statistics.median(v_ranges):.2f}\n")
    print("  lambda   median range of lambda*V   moves change        escapes the band   of which late")
    for lam in LAMBDAS:
        changed = escaped = late_escaped = 0
        for p, v in zip(table, vs):
            band = p.values >= p.values.max() - eps
            pick = int(np.argmax(p.values + lam * v))
            if pick != p.teacher:
                changed += 1
            if not band[pick]:
                escaped += 1
                if p.step >= LATE_STEP:
                    late_escaped += 1
        late = f"{late_escaped / n_late * 100:5.1f}%" if n_late else "    -"
        print(
            f"  {lam:<8.3f}{lam * statistics.median(v_ranges):>18.2f}"
            f"      {changed:>5}/{n} ({changed / n * 100:5.1f}%)"
            f"   {escaped:>5}/{n} ({escaped / n * 100:5.1f}%)   {late}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "value_h100.npz",
        help="V written by train_value.py --save (only the band diagnostics if absent)",
    )
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--steps", type=int, default=240)
    # Match the third-ply expectation measurement (early game included, every third move). V includes terms divided
    # by the fruit count, so looking only at the late game misses early-game effects.
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.1, help="width of the tie band")
    parser.add_argument(
        "--positions",
        type=Path,
        # A different name from the old format without boards (value_escape_positions.pkl).
        default=ROOT / "artifacts" / "value_escape_boards.pkl",
        help="cache of candidate tables. Reused if present.",
    )
    args = parser.parse_args()

    if args.positions.exists():
        table = pickle.loads(args.positions.read_bytes())
        if table and not isinstance(table[0], Position):
            raise SystemExit(
                f"{args.positions} is the old format without boards. "
                "Delete it or pass another name with --positions and rerun"
            )
        print(f"reusing cache {args.positions}")
    else:
        seeds = [args.seed + i for i in range(args.seeds)]
        table = _collect(seeds, args.steps, args.skip, args.stride)
        args.positions.parent.mkdir(parents=True, exist_ok=True)
        args.positions.write_bytes(pickle.dumps(table))
        print(f"saved cache: {args.positions}")

    if not table:
        raise SystemExit("no positions collected")
    feats = [
        np.stack([board_features(board, sign=p.sign) for board in p.boards])
        for p in table
    ]
    _report_band(table, feats, args.eps)
    if args.model.exists():
        print(f"\nV: {args.model}")
        _sweep(table, feats, load_value(args.model), args.eps)
    else:
        print(f"\nno V, so the λ sweep was skipped: {args.model}")
    print(
        "\nPositions where the band collapses to one board, and features that do not split inside the band,"
        "\ncannot be ordered by any weight. If only a few % escape the band, an A/B will not move score."
    )


if __name__ == "__main__":
    main()
