"""A screen before the A/B: does a weight change move moves outside the **two-ply** band?

`band_escape.py` cuts the band on the first-ply eval, and its table (NOTES 'The existing weights have
no leverage') was measured when only 2 held candidates got the next lookahead. The current policy ranks
`HELD_TOP` candidates by held eval + `NEXT_DISCOUNT` × best next, and a weight acts on both plies:
it changes which held candidates reach the top, and how every next reply is scored.
So here the weight is swept through the whole two-ply decision.

The physics runs one pass. Every held candidate keeps its per-term breakdown, and **every** living held
board keeps the breakdown of every next reply, so a multiplier that pulls a candidate from far down into
the top is still covered. The sweep is then analytic (seconds with the cache).

`--ply held|next|both` applies the multiplier to one ply only. A term that is right about the board but
wrong about a reply that has not been played yet shows up as a gap between those rows.

Usage:
  python scripts/weight_escape.py
  python scripts/weight_escape.py --eps 0.5
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT
from scripts.band_escape import SWEEP_KEYS, _components, _held_components
from src import policy as pol
from src.observe import Observation, clamp_drop_x
from src.policy import choose_x
from src.reward import is_lost
from src.sim.sim_env import SimEnv
from src.vision.classify import fruit_radius

# Wider than band_escape: under two plies a term also has to outweigh the discounted reply,
# so 0.5-2x may sit inside the band where 0.25 or 4x does not.
MULTIPLIERS = (0.0, 0.25, 0.5, 2.0, 4.0)
# `next` is not a term: it scales `NEXT_DISCOUNT`, the weight of the whole reply (13 of the avoidable
# size-order breaks were decided by it; NOTES 'What decides size-order breaks'). Only `--ply both` applies.
KEYS = ("score", *SWEEP_KEYS, "valley_grow", "next")
LATE_STEP = 60

Parts = dict[str, float]


@dataclass
class Position:
    seed: int
    step: int
    held: list[tuple[float, Parts, bool]]  # (eval, breakdown, dies) in rank_candidates order
    replies: list[list[tuple[float, Parts]] | None]  # per held candidate; None for dying ones
    has_next: bool
    teacher: int  # index into held chosen by choose_x


def _position(obs: Observation, seed: int, step: int) -> Position | None:
    assert obs.held_type is not None
    held_r = fruit_radius(obs.held_type)
    before = list(obs.fruits)
    ranked = pol.rank_candidates(obs)
    if not ranked:
        return None
    x = choose_x(obs, ranked=ranked)

    held: list[tuple[float, Parts, bool]] = []
    replies: list[list[tuple[float, Parts]] | None] = []
    all_dead = all(is_lost(after) for _e, _x, after, _s in ranked)
    for held_eval, hx, after, _score in ranked:
        parts, total = _held_components(before, obs.held_type, hx, held_r, obs.next_type)
        # Aggregating while the breakdown is off breaks the definition of the band, so stop.
        if abs(total - held_eval) > 1e-6:
            raise SystemExit(f"held breakdown does not match eval: {total} != {held_eval}")
        dies = is_lost(after)
        held.append((total, parts, dies))
        if obs.next_type is None or (dies and not all_dead):
            replies.append(None)
            continue
        next_r = fruit_radius(obs.next_type)
        rows: list[tuple[float, Parts]] = []
        nxs = [
            clamp_drop_x(nx, obs.next_type)
            for nx in pol._candidates(after, obs.next_type, next_r, step=pol.NEXT_CANDIDATE_STEP)
        ]
        for nx in nxs:
            # next is scored without the draw after it, as in policy._next_eval_job.
            reply_parts, reply_total = _components(after, obs.next_type, nx, next_r, None)
            rows.append((reply_total, reply_parts))
        if rows:
            ref = pol._next_eval_job(after, obs.next_type, next_r, nxs[0])
            if abs(rows[0][0] - ref) > 1e-6:
                raise SystemExit(f"next breakdown does not match eval: {rows[0][0]} != {ref}")
        replies.append(rows)

    pos = Position(seed, step, held, replies, obs.next_type is not None, -1)
    pick = _choose(pos, "score", 1.0, "both")
    xs = [hx for _e, hx, _a, _s in ranked]
    if pick is None or xs[pick] != x:
        raise SystemExit(f"replayed two-ply choice differs from choose_x at seed {seed} step {step}")
    pos.teacher = pick
    return pos


def _values(pos: Position, key: str, mult: float, ply: str) -> dict[int, float]:
    """Two-ply value of each candidate in the top, after scaling one term. The same rules as choose_x."""
    discount = pol.NEXT_DISCOUNT
    if key == "next":
        discount *= mult
        mult = 1.0
    held_mult = mult if ply in ("held", "both") else 1.0
    next_mult = mult if ply in ("next", "both") else 1.0
    evals = [total + parts.get(key, 0.0) * (held_mult - 1.0) for total, parts, _d in pos.held]
    # rank_candidates sorts by eval with a stable sort; held is already in the unscaled order.
    order = sorted(range(len(evals)), key=lambda i: evals[i], reverse=True)
    alive = [i for i in order if not pos.held[i][2]]
    top = (alive or order)[: pol.HELD_TOP]
    values = {}
    for i in top:
        best = 0.0
        if pos.has_next:
            rows = pos.replies[i]
            if rows is None:
                raise SystemExit("a dying candidate reached the top while living ones exist")
            if rows:
                best = max(t + p.get(key, 0.0) * (next_mult - 1.0) for t, p in rows)
        values[i] = evals[i] + discount * best
    return values


def _choose(pos: Position, key: str, mult: float, ply: str) -> int | None:
    values = _values(pos, key, mult, ply)
    if not values:
        return None
    best_i, best_v = None, float("-inf")
    for i, v in values.items():  # insertion order = rank order; strict > keeps the first like choose_x
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def _collect_seed(seed: int, steps: int, skip: int, stride: int) -> list[Position]:
    table: list[Position] = []
    env = SimEnv(seed=seed)
    obs = env.reset()
    for step in range(steps):
        if obs.held_type is None:
            break
        if step >= skip and step % stride == 0 and obs.fruits:
            row = _position(obs, seed, step)
            if row is not None:
                table.append(row)
        result = env.step(choose_x(obs))
        obs = result.observation
        if result.done:
            break
    return table


def _collect(
    seeds: list[int], steps: int, skip: int, stride: int, workers: int
) -> list[Position]:
    """One process per seed. Seeds are independent games, so the table equals the serial one in seed order."""
    table: list[Position] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for seed, rows in zip(
            seeds,
            pool.map(_collect_seed, seeds, *zip(*[(steps, skip, stride)] * len(seeds))),
        ):
            table.extend(rows)
            print(f"  seed {seed}: positions {len(rows)}", flush=True)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--steps", type=int, default=240)
    # Early game included, like value_escape: terms divided by the fruit count work early.
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.1, help="width of the tie band")
    parser.add_argument("--ply", choices=("both", "held", "next", "all"), default="all")
    parser.add_argument("--workers", type=int, default=1, help="processes for collection, one seed each")
    parser.add_argument(
        "--positions",
        type=Path,
        default=ROOT / "artifacts" / "weight_escape_positions.pkl",
        help="cache of candidate and reply breakdowns. Reused if present.",
    )
    args = parser.parse_args()

    if args.positions.exists():
        table = pickle.loads(args.positions.read_bytes())
        print(f"reusing cache {args.positions}")
    else:
        seeds = [args.seed + i for i in range(args.seeds)]
        table = _collect(seeds, args.steps, args.skip, args.stride, args.workers)
        args.positions.parent.mkdir(parents=True, exist_ok=True)
        args.positions.write_bytes(pickle.dumps(table))
        print(f"saved cache: {args.positions}")

    n = len(table)
    late = sum(1 for p in table if p.step >= LATE_STEP)
    bands = []
    for p in table:
        values = _values(p, "score", 1.0, "both")
        best = max(values.values())
        bands.append({i for i, v in values.items() if v >= best - args.eps})
    print(f"\n{n} positions ({late} late, step>={LATE_STEP})   two-ply band (eps={args.eps}) "
          f"median size {statistics.median(len(b) for b in bands):.0f}\n")
    plies = ("both", "held", "next") if args.ply == "all" else (args.ply,)
    print("term               ply   mult   moves change   escapes the band   of which late")
    for key in KEYS:
        for ply in plies if key != "next" else ("both",):
            for mult in MULTIPLIERS:
                changed = escaped = escaped_late = 0
                for p, band in zip(table, bands):
                    pick = _choose(p, key, mult, ply)
                    if pick != p.teacher:
                        changed += 1
                    if pick not in band:
                        escaped += 1
                        if p.step >= LATE_STEP:
                            escaped_late += 1
                print(f"  {key:<17}{ply:<6}x{mult:<5.2f}{changed / n * 100:6.1f}%"
                      f"        {escaped / n * 100:6.1f}%          {escaped_late / max(1, late) * 100:5.1f}%")
    print("\nOnly the fraction escaping the band can move score (NOTES 'Settled: the tie band really is indifferent')."
          "\nA few % will not show in an A/B at n=50.")


if __name__ == "__main__":
    main()
