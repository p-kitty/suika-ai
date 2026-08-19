"""A diagnostic trace counting fruits that sit on the board without merging (fossils).

The late game after the board breaks down is not 'it can no longer merge' but 'merging cannot keep up
with supply' (NOTES 'Investigated: sudden death from scattered low-tier fruits late in the game'). To see where that supply-side hole
is, each fruit gets an id and **the number of moves until it disappears** is measured.

Per move, the post-drop board is matched to the pre-drop board by nearest neighbor (same type, movement up to 3x the radius).
Unmatched fruits are considered new (dropped, or born from a merge). For fruits that stay,
the gap above them is classified too:

    touch  gap <= under.radius*0.6  … inside the window `_bury_penalty` counts
    near   gap <= under.radius*2    … a same-type partner (diameter 2r) cannot enter from above
    far    more than that           … open above; what blocks it is sideways
    none   no bigger fruit above

Matching looks only at type and distance, so if a same-type fruit is already near a fruit born from a merge
the ids can swap. When relying on an individual age, check the board by eye.
It does not matter when read as a distribution.

**The numbers here are diagnostics and not a basis for choosing weights.** Rules whose weights were chosen
by screening with home-made structural metrics not validated against score lost all three
A/Bs (NOTES 'How to measure'). It is used only as far as seeing
'which shapes are being missed'.

Usage:
  python scripts/fossils.py --seed 642746
  python scripts/fossils.py --seeds 5 --seed 910000
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import penalties as pen
from src.policy import choose_x
from src.sim.sim_env import SimEnv
from src.vision.colors import FRUIT_NAMES
from src.vision.state import Fruit

# Fruits staying on the board longer than this are called 'staying'. The measured median age is 3 moves, so
# 50 moves is 16x the median. A place to cut the tail of the distribution, not a meaningful threshold.
FOSSIL_AGE = 50
# Matching tolerance. Even when knocked, a fruit does not fly from one side of the board to the other in one move.
MATCH_RADII = 3.0


class _Tracker:
    """Match after to before per move, keeping the lifetime of each fruit."""

    def __init__(self) -> None:
        self._next_id = 0
        self.ids: list[int] = []
        self.birth: dict[int, int] = {}
        self.death: dict[int, int] = {}
        self.kind: dict[int, int] = {}

    def update(self, before: list[Fruit], after: list[Fruit], move: int) -> None:
        used: set[int] = set()
        new_ids: list[int] = []
        for fruit in after:
            best, best_d = None, math.inf
            for i, old in enumerate(before):
                if i in used or old.type != fruit.type:
                    continue
                d = abs(old.x - fruit.x) + abs(old.y - fruit.y)
                if d < best_d:
                    best, best_d = i, d
            if best is not None and best_d <= fruit.radius * MATCH_RADII:
                used.add(best)
                new_ids.append(self.ids[best])
                continue
            self.birth[self._next_id] = move
            self.kind[self._next_id] = fruit.type
            new_ids.append(self._next_id)
            self._next_id += 1
        for i, fid in enumerate(self.ids):
            if i not in used:
                self.death[fid] = move
        self.ids = new_ids

    def ages(self, last_move: int) -> list[tuple[int, int]]:
        """(moves survived, id). A fruit that never merged counts as alive until the last move."""
        return sorted(
            ((self.death.get(fid, last_move + 1) - birth, fid) for fid, birth in self.birth.items()),
            reverse=True,
        )


def _roof_gap(fruits: list[Fruit], under: Fruit) -> float:
    """Minimum gap to a bigger fruit directly above (the horizontal window is the same as `_bury_penalty`)."""
    best = math.inf
    for over in fruits:
        if over is under or over.type <= under.type:
            continue
        if over.y >= under.y:
            continue
        if abs(over.x - under.x) > (under.radius + over.radius) * 0.9:
            continue
        gap = (under.y - under.radius) - (over.y + over.radius)
        if gap < -pen.MERGE_SLACK:
            continue
        best = min(best, gap)
    return best


def _roof_bucket(fruits: list[Fruit], fruit: Fruit) -> int:
    gap = _roof_gap(fruits, fruit)
    if gap == math.inf:
        return 3
    if gap <= fruit.radius * 0.6:
        return 0
    if gap <= fruit.radius * 2.0:
        return 1
    return 2


def _play(seed: int, max_steps: int) -> tuple[_Tracker, dict[int, list[int]], int]:
    """Play one game through and return the tracker and [moves, touch, near, far, none, partner present] per id."""
    env = SimEnv(seed=seed)
    obs = env.reset()
    tracker = _Tracker()
    stat: dict[int, list[int]] = {}
    last_move = 0
    for move in range(1, max_steps + 1):
        before = list(obs.fruits)
        result = env.step(choose_x(obs))
        after = list(result.observation.fruits)
        tracker.update(before, after, move)
        for fid, fruit in zip(tracker.ids, after):
            row = stat.setdefault(fid, [0, 0, 0, 0, 0, 0])
            row[0] += 1
            row[1 + _roof_bucket(after, fruit)] += 1
            row[5] += int(sum(1 for f in after if f.type == fruit.type) >= 2)
        last_move = move
        if result.done:
            break
        obs = result.observation
    return tracker, stat, last_move


def _report(seed: int, tracker: _Tracker, stat: dict[int, list[int]], last_move: int) -> None:
    ages = tracker.ages(last_move)
    values = sorted(age for age, _fid in ages)
    fossils = [(age, fid) for age, fid in ages if age >= FOSSIL_AGE]
    print(
        f"seed={seed} {last_move} moves  {len(ages)} fruits born, "
        f"age median {statistics.median(values):.0f} / mean {statistics.mean(values):.1f} / "
        f"max {values[-1]}  staying ({FOSSIL_AGE}+ moves) {len(fossils)} "
        f"({len(fossils) / len(ages):.1%})"
    )
    if fossils:
        print(f"  {'type':<11}{'age':>5}{'touch':>8}{'near':>8}{'far':>8}{'none':>8}{'partner':>8}")
        for age, fid in fossils:
            n, touch, near, far, none, pair = stat[fid]
            print(
                f"  {FRUIT_NAMES[tracker.kind[fid]]:<11}{age:5d}{touch / n:8.1%}"
                f"{near / n:8.1%}{far / n:8.1%}{none / n:8.1%}{pair / n:8.1%}"
            )


def _bands(tracker: _Tracker, stat: dict[int, list[int]], last_move: int) -> dict[str, list[int]]:
    """Total fruit-moves per age band and the breakdown above them (for summing across seeds)."""
    out: dict[str, list[int]] = {}
    for lo, hi, label in ((0, 5, "0-5"), (5, 20, "5-20"), (20, 50, "20-50"), (50, 10**9, "50+")):
        row = out.setdefault(label, [0, 0, 0, 0, 0, 0])
        for age, fid in tracker.ages(last_move):
            if not lo <= age < hi:
                continue
            for i in range(6):
                row[i] += stat[fid][i]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=642746)
    parser.add_argument("--seeds", type=int, default=1, help="number of consecutive seeds to run from seed")
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()

    total: dict[str, list[int]] = {}
    for i in range(args.seeds):
        seed = args.seed + i
        tracker, stat, last_move = _play(seed, args.max_steps)
        _report(seed, tracker, stat, last_move)
        for label, row in _bands(tracker, stat, last_move).items():
            acc = total.setdefault(label, [0, 0, 0, 0, 0, 0])
            for j in range(6):
                acc[j] += row[j]
        print(flush=True)

    print(f"=== by age band, in fruit-moves (sum of {args.seeds} games) ===")
    print(f"  {'age':<8}{'total':>7}{'touch':>8}{'near':>8}{'far':>8}{'none':>8}{'partner':>8}")
    for label in ("0-5", "5-20", "20-50", "50+"):
        row = total.get(label)
        if not row or not row[0]:
            continue
        n = row[0]
        print(
            f"  {label:<8}{n:7d}{row[1] / n:8.1%}{row[2] / n:8.1%}"
            f"{row[3] / n:8.1%}{row[4] / n:8.1%}{row[5] / n:8.1%}"
        )


if __name__ == "__main__":
    main()
