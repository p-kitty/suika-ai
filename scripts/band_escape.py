"""重みの変更が「同点帯の外」へ手を動かすかを測る、A/B の前の足切り。

同点帯 (最良から `--eps` 以内の候補) の中は無差別だと n=133 で確かめてある
(NOTES「決着: 同点帯は本当に無差別」)。帯の中で手が変わっても score は動かない。
なので「手が変わる割合」は足切りにならず、**帯の外へ出た割合**を見る。

予測力は既知の負け 2 件で検証済み。`drop_ideal` は手を 50.2% 変えるのに
帯の外へ出るのは 6.3%、凸凹 x4 は 17.5% に対し 7.0% で、どちらも
n=133 の A/B で score が動かなかった。

物理は 1 パスだけ回して各候補の項の内訳を持ち、重みの掃引はそこから
解析的にやる (再実行は `--positions` のキャッシュが効くので数秒)。

用法:
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

# 帯を測るのは終盤。序盤の盤は実が少なく、梯子も詰まりも成立していない。
DEFAULT_SKIP = 60
# 掃引する倍率。1.0 は恒等なので入れない。
MULTIPLIERS = (0.5, 1.5, 2.0)
# 倍率で掃引できる項 (重み付きの寄与をそのまま定数倍する)。
SWEEP_KEYS = ("bury", "excess_same", "size_order", "big_layout", "foreign_aim", "variance")


def _components(
    before: list, drop_type: int, x: float, held_r: float, next_type: int | None
) -> tuple[dict[str, float], float]:
    """1 候補ぶんの (項名 -> eval への寄与, eval)。

    `policy._evaluate_drop` と同じ分解。合計が `_held_eval` と一致することは
    呼び元が局面ごとに突き合わせる (ずれたまま集計すると帯の定義が狂う)。
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
        "packed": 0.0,
        "valley_grow": 0.0,
    }
    if not held_merged:
        parts["packed"] = -pen.packed_small_side_penalty(
            before, land_x, drop_type, held_r, sign
        )
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
        print(f"  seed {seed}: 局面 {len(positions)}", flush=True)
    return positions


def _candidate_table(positions: list[Observation]) -> list[list[tuple[float, dict[str, float]]]]:
    """各局面の全候補の (eval, 項の内訳)。物理を回すのはここだけ。"""
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
        # 分解が実際の eval と一致しているか、局面ごとに 1 本だけ突き合わせる。
        # ずれたまま集計すると帯の定義そのものが狂うので、黙って進めない。
        if first_x is not None:
            _after, ref = pol._held_eval(obs, first_x, held_r)
            if abs(rows[0][0] - ref) > 1e-6:
                raise SystemExit(f"分解が eval と一致しない: {rows[0][0]} != {ref}")
        table.append(rows)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(positions)}", flush=True)
    return table


def _escape(table, key: str, mult: float, eps: float) -> tuple[int, int]:
    """(手が変わった局面数, 帯の外へ出た局面数)。"""
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
    parser.add_argument("--eps", type=float, default=0.1, help="同点帯の幅")
    parser.add_argument(
        "--positions",
        type=Path,
        default=ROOT / "artifacts" / "band_positions.pkl",
        help="局面と候補表のキャッシュ。あれば再利用する。",
    )
    args = parser.parse_args()

    if args.positions.exists():
        table = pickle.loads(args.positions.read_bytes())
        print(f"キャッシュ {args.positions} を再利用")
    else:
        seeds = [args.seed + i for i in range(args.seeds)]
        positions = _collect(seeds, args.steps, args.skip, args.stride)
        table = _candidate_table(positions)
        args.positions.parent.mkdir(parents=True, exist_ok=True)
        args.positions.write_bytes(pickle.dumps(table))
        print(f"キャッシュを保存: {args.positions}")

    n = len(table)
    sizes = [
        sum(1 for total, _p in rows if total >= max(t for t, _q in rows) - args.eps)
        for rows in table
    ]
    print(f"\n局面 {n} 件   帯 (eps={args.eps}) の候補数 中央 {statistics.median(sizes):.0f}\n")
    print("項            倍率    手が変わる         帯の外へ出る")
    for key in SWEEP_KEYS:
        for mult in MULTIPLIERS:
            changed, escaped = _escape(table, key, mult, args.eps)
            print(
                f"  {key:<12}x{mult:<4.1f}{changed:>5}/{n} ({changed / n * 100:5.1f}%)"
                f"   {escaped:>5}/{n} ({escaped / n * 100:5.1f}%)"
            )


if __name__ == "__main__":
    main()
