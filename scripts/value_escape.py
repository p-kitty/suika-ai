"""A screen before the A/B: does adding the learned V move moves outside 'the two-ply band'?

A different role from `band_escape.py`. That one cuts the band on the **first-ply eval**, but the current policy
ranks `HELD_TOP` candidates by the two-ply value (held eval + `NEXT_DISCOUNT` × best next),
so looking at the first-ply band does not show the current policy's band
(→NOTES 'Measured and dropped: third-ply expectation'). Here the same 8 are
reordered by Q = two-ply value + λ·V(post-drop board), reporting **the fraction escaping the band** per λ.

Escaping the band is only a necessary condition (`crown_danger`, which escaped 12.4%, was also
null at n=250). If a few %, running the A/B will not move score.

The physics runs one pass keeping each candidate's (two-ply value, feature vector), and the λ sweep is done
analytically from that. **It keeps features, so refitting V needs no replay**
(seconds thanks to the `--positions` cache).

Usage:
  python scripts/value_escape.py --model artifacts/value_h100.npz
  python scripts/value_escape.py --model artifacts/value_h100.npz --eps 0.5
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT
from src import policy as pol
from src.policy import choose_x, rank_candidates
from src.reward import is_lost
from src.sim.sim_env import SimEnv
from src.training.features import board_features
from src.training.value import LinearValue, load as load_value

# λ values swept. V's candidate range is 3 orders of magnitude larger than the eval band width (0.1),
# so unless swept in 10x steps it jumps between 'no effect' and 'V takes over'.
LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _position_table(obs) -> tuple[np.ndarray, np.ndarray, int] | None:
    """(two-ply values (K,), features (K, D), index chosen by the teacher) for one position.

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
    sign = pol._order_sign(list(obs.fruits))
    feats = np.stack([board_features(board, sign=sign) for board in boards])

    # Check per position that the breakdown matches the actual move. Aggregating while it is off
    # breaks the definition of the band itself, so do not proceed silently.
    teacher = int(np.argmax(values))
    if top[teacher][1] != x:
        raise SystemExit(f"two-ply argmax differs from choose_x: {top[teacher][1]} != {x}")
    return values, feats, teacher


def _collect(
    seeds: list[int], steps: int, skip: int, stride: int
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    table: list[tuple[np.ndarray, np.ndarray, int]] = []
    for seed in seeds:
        env = SimEnv(seed=seed)
        obs = env.reset()
        for step in range(steps):
            if obs.held_type is None:
                break
            if step >= skip and step % stride == 0:
                row = _position_table(obs)
                if row is not None:
                    table.append(row)
            result = env.step(choose_x(obs))
            obs = result.observation
            if result.done:
                break
        print(f"  seed {seed}: positions {len(table)}", flush=True)
    return table


def _sweep(
    table: list[tuple[np.ndarray, np.ndarray, int]], model: LinearValue, eps: float
) -> None:
    vs = [model.predict(feats) for _values, feats, _teacher in table]
    n = len(table)

    band_sizes = [int((v >= v.max() - eps).sum()) for v, _f, _t in table]
    v_ranges = [float(v.max() - v.min()) for v in vs]
    all_tied = sum(1 for (values, _f, _t), size in zip(table, band_sizes) if size == len(values))
    print(f"\n{n} positions   candidates (HELD_TOP) median {statistics.median(len(v) for v, _f, _t in table):.0f}")
    print(
        f"two-ply band (eps={eps}) candidate count median {statistics.median(band_sizes):.0f}"
        f"   all candidates in the band {all_tied}/{n} ({all_tied / n * 100:.1f}%)"
    )
    print(f"V candidate range median {statistics.median(v_ranges):.2f}\n")

    print("  lambda   median range of lambda*V   moves change        escapes the band")
    for lam in LAMBDAS:
        changed = escaped = 0
        for (values, _feats, teacher), v in zip(table, vs):
            q = values + lam * v
            band = values >= values.max() - eps
            pick = int(np.argmax(q))
            if pick != teacher:
                changed += 1
            if not band[pick]:
                escaped += 1
        print(
            f"  {lam:<8.3f}{lam * statistics.median(v_ranges):>18.2f}"
            f"      {changed:>5}/{n} ({changed / n * 100:5.1f}%)"
            f"   {escaped:>5}/{n} ({escaped / n * 100:5.1f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "value_h100.npz",
        help="V written by train_value.py --save",
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
        default=ROOT / "artifacts" / "value_escape_positions.pkl",
        help="cache of candidate tables. Reused if present.",
    )
    args = parser.parse_args()

    if args.positions.exists():
        table = pickle.loads(args.positions.read_bytes())
        print(f"reusing cache {args.positions}")
    else:
        seeds = [args.seed + i for i in range(args.seeds)]
        table = _collect(seeds, args.steps, args.skip, args.stride)
        args.positions.parent.mkdir(parents=True, exist_ok=True)
        args.positions.write_bytes(pickle.dumps(table))
        print(f"saved cache: {args.positions}")

    if not table:
        raise SystemExit("no positions collected")
    print(f"V: {args.model}")
    _sweep(table, load_value(args.model), args.eps)
    print(
        "\nIf only a few % escape the band, running an A/B will not move score."
        "\nConversely, a λ that escapes close to 100% means V has taken over eval,"
        "\nthe same as discarding the individually validated penalties wholesale."
    )


if __name__ == "__main__":
    main()
