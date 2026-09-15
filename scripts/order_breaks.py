"""Diagnostic trace: moves where the dropped fruit itself breaks the left-right size order, and why they won.

Plays the current policy and, on every move where held does not merge, counts the inversions **the landed
fruit itself** creates (pairs with it, the same pair and valley-exemption rules as `_size_order_penalty`,
summed type gap). A global inversion count does not move with where one fruit goes (NOTES 'Ideas that did
not work'), so only this local count is looked at.

For each flagged move the best *clean* candidate is compared with the
chosen move on the two-ply value `choose_x` ranks by, and the gap is split into two kinds, whose remedies
differ completely (AGENTS 'Do not run an A/B while obvious blunders remain'):

A move is *clean* when held merges or its landed fruit creates less than `--min-gap`.

- **tie** … the clean move is inside the band (gap <= --eps). No weight fixes this; ignore it
- **decided** … the clean move lost by more. The term that contributed most to the gap is named;
  suspect that term's firing condition, not its weight. `next` means the discounted best reply decided it

**The counts here are diagnostics, not a basis for choosing weights** (NOTES 'How to measure':
rules tuned on home-made structural metrics lost all three A/Bs). Use them to pick positions to
replay with `view_sim.py --seed <seed>` at the printed move.

Usage:
  python scripts/order_breaks.py
  python scripts/order_breaks.py --seeds 2 --seed 642746 --min-gap 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.band_escape import _held_components
from src import penalties as pen
from src import policy as pol
from src.observe import Observation
from src.policy import choose_x
from src.reward import is_lost
from src.sim.sim_env import SimEnv
from src.sim.sim_physics import simulate_drop_held
from src.vision.classify import fruit_radius
from src.vision.colors import FRUIT_NAMES
from src.vision.state import Fruit

# How many clean candidates (by first-ply eval) get the two-ply value. The lookahead is the heavy part,
# and a clean move ranked below this on the first ply is far outside the band anyway.
CLEAN_TOP = 3


@dataclass
class _Break:
    inv: float
    kind: str
    gap: float = float("nan")
    term: str = "-"
    clean_x: float = float("nan")
    term_diff: float = float("nan")


def _local_inversion(held: Fruit, fruits: list[Fruit], sign: int) -> float:
    """Summed type gap of inverted pairs that include held, with the rules of `_size_order_penalty`."""
    if pen._size_order_exempt(held, fruits):
        return 0.0
    total = 0.0
    for f in fruits:
        if f is held or f.type == held.type:
            continue
        if abs(f.x - held.x) < min(f.radius, held.radius) * 0.5:
            continue
        if pen._size_order_exempt(f, fruits):
            continue
        left, right = (f, held) if f.x <= held.x else (held, f)
        if sign > 0 and left.type < right.type:
            total += right.type - left.type
        elif sign < 0 and left.type > right.type:
            total += left.type - right.type
    return total


def _drop_inversion(before: list[Fruit], drop_type: int, x: float, sign: int) -> tuple[bool, float]:
    """(held merged, local inversion of the landed fruit)."""
    after, _m, _types, held_merged, held_fruit = simulate_drop_held(before, drop_type, x)
    if held_merged or held_fruit is None:
        return True, 0.0
    landed = next((f for f in after if f == held_fruit), None)
    if landed is None:
        return True, 0.0
    return False, _local_inversion(landed, after, sign)


def _two_ply(obs: Observation, x: float, held_r: float) -> tuple[float, float]:
    """(two-ply value, discounted best reply) of one held column, as choose_x scores it."""
    held_eval, _x, after, _score = pol._held_eval_job(obs, held_r, x)
    reply = 0.0
    if obs.next_type is not None:
        reply = pol._best_next_scores([after], obs.next_type, step=pol.NEXT_CANDIDATE_STEP)[0]
    return held_eval + pol.NEXT_DISCOUNT * reply, pol.NEXT_DISCOUNT * reply


def _inspect(obs: Observation, x: float, min_gap: float, eps: float) -> _Break | None:
    assert obs.held_type is not None
    before = list(obs.fruits)
    sign = pol._order_sign(before)
    merged, inv = _drop_inversion(before, obs.held_type, x, sign)
    if merged or inv < min_gap:
        return None

    held_r = fruit_radius(obs.held_type)
    ranked = pol.rank_candidates(obs)
    alive = [row for row in ranked if not is_lost(row[2])] or ranked
    clean = []
    for _eval, cx, _after, _score in alive:
        c_merged, c_inv = _drop_inversion(before, obs.held_type, cx, sign)
        if c_merged or c_inv < min_gap:
            clean.append(cx)
            if len(clean) >= CLEAN_TOP:
                break
    chosen_value, chosen_reply = _two_ply(obs, x, held_r)
    if not clean:
        return _Break(inv, "no clean move")

    scored = [(cx, *_two_ply(obs, cx, held_r)) for cx in clean]
    clean_x, clean_value, clean_reply = max(scored, key=lambda row: row[1])
    gap = chosen_value - clean_value
    if gap <= eps:
        return _Break(inv, "tie", gap, clean_x=clean_x)

    chosen_parts, _ = _held_components(before, obs.held_type, x, held_r, obs.next_type)
    clean_parts, _ = _held_components(before, obs.held_type, clean_x, held_r, obs.next_type)
    diffs = {k: chosen_parts[k] - clean_parts[k] for k in chosen_parts}
    diffs["next"] = chosen_reply - clean_reply
    term = max(diffs, key=lambda k: diffs[k])
    return _Break(inv, "decided", gap, term, clean_x, diffs[term])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--min-gap", type=float, default=2.0,
                        help="summed type gap of the inversions the landed fruit creates to flag a move")
    parser.add_argument("--eps", type=float, default=0.1, help="width of the tie band")
    args = parser.parse_args()

    kinds: Counter[str] = Counter()
    terms: dict[str, list[float]] = defaultdict(list)
    quiet_moves = 0
    for seed in range(args.seed, args.seed + args.seeds):
        env = SimEnv(seed=seed)
        obs = env.reset()
        for i in range(args.steps):
            if obs.held_type is None:
                break
            x = choose_x(obs)
            row = _inspect(obs, x, args.min_gap, args.eps)
            if row is not None:
                kinds[row.kind] += 1
                if row.kind == "decided":
                    terms[row.term].append(row.gap)
                    print(f"  seed {seed} move {i + 1:>3}  {FRUIT_NAMES[obs.held_type]:<10} "
                          f"x={x:5.1f} inv={row.inv:.0f}  clean x={row.clean_x:5.1f} "
                          f"loses by {row.gap:6.2f}  to {row.term} (+{row.term_diff:.1f})",
                          flush=True)
            result = env.step(x)
            if not result.merges:
                quiet_moves += 1
            obs = result.observation
            if result.done:
                break
        print(f"seed {seed}: done at move {i + 1}", flush=True)

    flagged = sum(kinds.values())
    print(f"\nmoves without a merge {quiet_moves}, flagged (inv >= {args.min_gap:g}) {flagged}")
    for kind, count in kinds.most_common():
        print(f"  {kind:<14}{count:>5}  ({count / max(1, flagged) * 100:.0f}%)")
    print("\ndecided by   count   median gap")
    for term, gaps in sorted(terms.items(), key=lambda kv: -len(kv[1])):
        print(f"  {term:<13}{len(gaps):>4}   {statistics.median(gaps):8.2f}")


if __name__ == "__main__":
    main()
