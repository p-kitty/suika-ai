"""bootstrap の 2 変種を対戦させ、差をフェーズ別に出す。

変更を入れたあと「どこが良く/悪くなったか」を局在させるための道具。
既定は床埋め後の置き分け (SUIKA_PACKED) の ON/OFF 比較。同じシード列を両方に流し、
平均だけでなくシードごとの勝ち負けと、序盤・終盤に分けた指標を出す。

--seed を省略すると毎回ランダムなシードから始める。固定シードを何度も
使い回すと、たまたまそのシード集合で起きた崩れを「再現した」と誤読しやすい。

用法:
  python scripts/compare_policy.py
  python scripts/compare_policy.py --episodes 60 --max-steps 300 --workers 8
"""

from __future__ import annotations

import argparse
import secrets
import statistics
import sys
import time
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parallel import default_workers
from src.reward import watermelon_count

# 序盤とみなす手数。ここまでの崩れ方を後半と分けて見る。
EARLY_STEPS = 30
# 連鎖発火とみなす 1 手あたりの合成数。
CASCADE_MERGES = 3


def _episode(
    seed: int,
    max_steps: int,
    variant: bool,
    pool: Executor | None = None,
) -> dict[str, float]:
    """1 エピソード回して指標を返す。ProcessPool のワーカー側で走る。

    pool は「エピソード並列の対象が無い (workers<=1)」ときだけ _run から渡される。
    エピソード自体を ProcessPool に分散する側では二重並列を避けるため None のまま。
    """
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
        # y は下向き。小さいほど山が高い = 危ない。
        "early_crown": min(early_crowns) if early_crowns else float("nan"),
        # 打ち切りに届かず死んだか。序盤崩壊の直接の指標。
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
    print(f"  {label}: {len(seeds)} エピソード実行中...", flush=True)
    if workers <= 1:
        # エピソード並列の対象が無いぶん、choose_x の候補評価を並列化する。
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
    print(f"  {label}: 完了 {time.monotonic() - started:.0f}s" + " " * 20, flush=True)
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="省略時は毎回ランダム。固定シードの使い回しは避けること。",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else default_workers()
    seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000)

    seeds = [seed + i for i in range(args.episodes)]
    base = _run(seeds, args.max_steps, False, workers, label="A (OFF)")
    new = _run(seeds, args.max_steps, True, workers, label="B (ON) ")

    print(
        f"\nepisodes={args.episodes} seed={seed} "
        f"max_steps={args.max_steps} workers={workers}"
    )
    print("  A = 床埋め後の置き分け OFF (現状)   B = ON")

    # 全部が打ち切りだと、伸びしろも死亡率も測れていない。
    # 仕込みのコストだけ見てリターンを見ない測定になるので、先に警告する。
    capped = sum(1 for row in base + new if row["steps"] >= args.max_steps)
    if capped == len(base) + len(new):
        print(
            f"\n  ** 全 {capped} エピソードが max_steps={args.max_steps} で打ち切り。**\n"
            "  ** 自然終了が 1 つも無く、生存時間も到達段階も測れていない。**\n"
            "  ** --max-steps を増やして測り直すこと。以下の数字は信用しない。**"
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

    # シードごとの対戦。平均が動かなくても勝ち負けが割れていれば別物。
    wins = sum(1 for a, b in zip(base, new) if b["score"] > a["score"])
    losses = sum(1 for a, b in zip(base, new) if b["score"] < a["score"])
    ties = args.episodes - wins - losses
    print(f"\n  seed 対戦 (B 視点)  win={wins}  loss={losses}  tie={ties}")
    if ties == args.episodes:
        print("  ** 全シードで同一。変更はこの条件では発火していない **")

    worst = sorted(zip(base, new), key=lambda p: p[1]["score"] - p[0]["score"])[:5]
    print("\n  悪化の大きいシード:")
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
