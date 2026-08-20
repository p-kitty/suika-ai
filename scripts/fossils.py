"""合体せず盤に居座る実（fossil）を数える、診断用のトレース。

盤が崩れたあとの終盤は「合体できなくなる」のではなく「合体しても供給に
追いつかない」（NOTES「終盤の低段位散在による即死」）。その供給側の穴が
どこにあるかを見るために、実 1 個ずつに id を振って**何手で消えたか**を測る。

手ごとに落下後の盤を落下前へ最近傍で対応付ける（同型・移動量が半径の 3 倍まで）。
対応が付かない実は新規（投下されたか、合体で生まれた）とみなす。居座った実に
ついては頭上の隙間も分類する:

    touch  隙間 <= under.radius*0.6  … `_bury_penalty` が数える窓の中
    near   隙間 <= under.radius*2    … 同種の相方（直径 2r）が上から入れない
    far    それ以上                   … 頭上は空いていて、塞いでいるのは横
    none   頭上に大きい実が無い

対応付けは型と距離だけで見るので、合体で生まれた実の近くに同型の実が既にいると
id が入れ替わることがある。1 個ずつの age を根拠にするときは盤を目で確かめる。
分布として読むぶんには効かない。

**ここで出る数字は診断であって、重みを決める根拠にはならない。** score で
検証していない自作の構造指標で足切りして重みを決めた規則は、A/B で 3 つとも
負けている（NOTES「測定のしかた」）。使うのは「どの形が取り逃されているか」を
見るところまで。

用法:
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

# これ以上盤に残った実を「居座り」と呼ぶ。実測の age 中央は 3 手なので、
# 50 手は中央の 16 倍。分布の裾を切る位置であって、意味のある閾値ではない。
FOSSIL_AGE = 50
# 対応付けの許容。弾かれても 1 手で盤の端から端へは飛ばない。
MATCH_RADII = 3.0


class _Tracker:
    """手ごとに after を before へ対応付け、実 1 個ずつの生存期間を持つ。"""

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
        """(生存手数, id)。合体しなかった実は最後の手まで生きたものとして数える。"""
        return sorted(
            ((self.death.get(fid, last_move + 1) - birth, fid) for fid, birth in self.birth.items()),
            reverse=True,
        )


def _roof_gap(fruits: list[Fruit], under: Fruit) -> float:
    """真上（横窓は `_bury_penalty` と同じ）にある大きい実との最小隙間。"""
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
    """1 局を通し、tracker と id ごとの [手数, touch, near, far, none, 相方あり] を返す。"""
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
        f"seed={seed} {last_move} 手  実 {len(ages)} 個生まれて "
        f"age 中央 {statistics.median(values):.0f} / 平均 {statistics.mean(values):.1f} / "
        f"最大 {values[-1]}  居座り({FOSSIL_AGE}手以上) {len(fossils)} 個 "
        f"({len(fossils) / len(ages):.1%})"
    )
    if fossils:
        print(f"  {'type':<11}{'age':>5}{'touch':>8}{'near':>8}{'far':>8}{'none':>8}{'相方':>8}")
        for age, fid in fossils:
            n, touch, near, far, none, pair = stat[fid]
            print(
                f"  {FRUIT_NAMES[tracker.kind[fid]]:<11}{age:5d}{touch / n:8.1%}"
                f"{near / n:8.1%}{far / n:8.1%}{none / n:8.1%}{pair / n:8.1%}"
            )


def _bands(tracker: _Tracker, stat: dict[int, list[int]], last_move: int) -> dict[str, list[int]]:
    """age 帯ごとの延べ手数と頭上の内訳（複数シードの合算用）。"""
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
    parser.add_argument("--seeds", type=int, default=1, help="seed から連番で回す本数")
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

    print(f"=== age 帯別・延べ手数ベース（{args.seeds} 局の合算） ===")
    print(f"  {'age':<8}{'延べ':>7}{'touch':>8}{'near':>8}{'far':>8}{'none':>8}{'相方':>8}")
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
