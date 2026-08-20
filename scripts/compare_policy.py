"""bootstrap の 2 変種を対戦させ、差をフェーズ別に出す。

変更を入れたあと「どこが良く/悪くなったか」を局在させるための道具。
同じシード列を A と B の両方に流し、平均だけでなくシードごとの勝ち負けと、
序盤・終盤に分けた指標を出す。

**比較したい変更は `_apply_variant` に差す。** ここが空のままだと A と B が
同じ方策になり、全シード tie の警告が出る。

指標ごとに、同一シードで対にした差の t 値と 95% CI を出す（`src/stats.py`）。
平均の増減だけを見て「良くなった」と読むのを防ぐためで、CI が 0 をまたぐ行は
この n では何も言えていない。n@5% 列はその指標を 5% の精度で語るのに要る
エピソード数で、値が小さい指標ほど少ない試行で変化を読める。

--seed を省略すると毎回ランダムなシードから始める。固定シードを何度も
使い回すと、たまたまそのシード集合で起きた崩れを「再現した」と誤読しやすい。

用法:
  python scripts/compare_policy.py
  python scripts/compare_policy.py --episodes 60 --max-steps 300 --workers 8
  python scripts/compare_policy.py --episodes 100 --max-steps 400 --out artifacts/ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.util.parallel import default_workers
from src.reward import watermelon_count
from src.util.stats import correlation, paired_stats

# 序盤とみなす手数。ここまでの崩れ方を後半と分けて見る。
EARLY_STEPS = 30
# 連鎖発火とみなす 1 手あたりの合成数。
CASCADE_MERGES = 3


def _apply_variant(enabled: bool) -> None:
    """比較したい変更をここで切り替える (B 側が enabled=True)。

    ワーカープロセスごとに、エピソードを回す前に呼ばれる。恒久化した規則の
    トグルを残しておくと死んだ分岐が増えるので、実験が終わったら中身は空に
    戻す。空のままだと A と B が同じ方策になり、全シード tie の警告が出る。

    重みはモジュール属性を書き換えて差し替える。減点側の定数は src.penalties、
    探索の粗さや先読み割引は src.policy にある。

    例:
        from src import penalties
        penalties.BURY_WEIGHT = 30.0 if enabled else 20.0

    いまは「その項が要るか」の除去実験。長い実行の最中に file を書き換えると
    走っている側に影響しうるので、どの項を切るかは環境変数 AB_ABLATE で選ぶ。
    """
    if not enabled:
        return
    from src import penalties

    term = os.environ.get("AB_ABLATE", "")
    if term == "variance":
        penalties.VARIANCE_WEIGHT = 0.0
    elif term == "excess_same":
        # 重み 20.0 は関数のローカルなので、関数ごと差し替えて切る。
        penalties._excess_same_penalty = lambda fruits: 0.0
    else:
        raise SystemExit(f"AB_ABLATE が未設定か不正: {term!r}")


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
    from src.policy import choose_x
    from src.sim.sim_env import SimEnv

    _apply_variant(variant)

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


def _column(rows: list[dict[str, float]], key: str) -> list[float]:
    return [r[key] for r in rows]


def _fmt(value: float, digits: int) -> str:
    return "-" if value != value else f"{value:.{digits}f}"


def _line(
    label: str,
    base: list[dict[str, float]],
    new: list[dict[str, float]],
    *,
    digits: int = 2,
) -> str:
    """1 指標ぶんの行。平均だけでなくペア差の t・95% CI・必要 n まで出す。

    n@5% は「その指標を 5% の精度で語るのに要るエピソード数」。指標どうしを
    同じものさしで比べるために相対値にしてある。これが小さい指標ほど、
    少ないエピソードで変化を読める＝代理指標に向く。
    r は同じ行の指標と score の相関で、代理指標として意味があるかの目安。
    """
    stats = paired_stats(_column(base, label), _column(new, label))
    pct = f"{stats.delta / stats.mean_a * 100:+6.1f}%" if stats.mean_a else "     -"
    ci = (
        f"[{stats.ci_lo:+8.1f},{stats.ci_hi:+8.1f}]"
        if stats.ci_lo == stats.ci_lo
        else " " * 19
    )
    n5 = stats.required_n(abs(stats.mean_a) * 0.05) if stats.mean_a else float("nan")
    r = correlation(
        _column(base, label) + _column(new, label),
        _column(base, "score") + _column(new, "score"),
    )
    mark = " *" if stats.significant else "  "
    return (
        f"  {label:<12}{_fmt(stats.mean_a, digits):>9} ->{_fmt(stats.mean_b, digits):>9}"
        f" ({pct}) t={_fmt(stats.t, 2):>6} {ci}{mark}"
        f" n@5%={_fmt(n5, 0):>7} r={_fmt(r, 2):>5}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="省略時は毎回ランダム。固定シードの使い回しは避けること。",
    )
    # 既定 100 は自然終了 (中央 210 手) の半分で、実測 200 本すべてが打ち切りだった。
    # 仕込みのコストだけ測ってリターンを測らない既定になっていたので 400 にする。
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="シードごとの生データを JSON で保存する（後から指標を選び直すため）。",
    )
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else default_workers()
    seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000)

    seeds = [seed + i for i in range(args.episodes)]
    base = _run(seeds, args.max_steps, False, workers, label="A (base)")
    new = _run(seeds, args.max_steps, True, workers, label="B (new) ")

    # 集計より先に保存する。長い実行の生データを、集計側の些細な不具合で失わない。
    if args.out is not None:
        # docstring は -OO で剥がれるので、無い前提で読む。ここで落ちると
        # 走り終わった数時間ぶんの生データを保存前に失う。
        variant_doc = (_apply_variant.__doc__ or "").strip().splitlines()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "episodes": args.episodes,
                    "max_steps": args.max_steps,
                    "variant": variant_doc[0] if variant_doc else "",
                    "a": base,
                    "b": new,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"  生データを保存: {args.out}")

    print(
        f"\nepisodes={args.episodes} seed={seed} "
        f"max_steps={args.max_steps} workers={workers}"
    )
    print("  A = _apply_variant(False)   B = _apply_variant(True)")

    # 打ち切りは伸びた側の頭を押さえるので、良い変更ほど過小評価される。
    # 「全部打ち切り」だけを警告していると、部分的な打ち切りが無警告で通ってしまう。
    total = len(base) + len(new)
    capped = sum(1 for row in base + new if row["steps"] >= args.max_steps)
    if capped == total:
        print(
            f"\n  ** 全 {capped} エピソードが max_steps={args.max_steps} で打ち切り。**\n"
            "  ** 自然終了が 1 つも無く、生存時間も到達段階も測れていない。**\n"
            "  ** --max-steps を増やして測り直すこと。以下の数字は信用しない。**"
        )
    elif capped:
        print(
            f"\n  ** {capped}/{total} エピソードが max_steps={args.max_steps} で打ち切り"
            f" ({capped / total * 100:.0f}%)。**\n"
            "  ** 打ち切られたのは伸びた対局なので、差はその分だけ圧縮されている。**"
        )
    print()
    for key, digits in (
        ("score", 2),
        ("early_score", 2),
        ("steps", 1),
        ("merges", 1),
        ("cascades", 2),
        ("max_type", 2),
        ("early_crown", 1),
        ("dead", 3),
        ("dead_early", 3),
    ):
        print(_line(key, base, new, digits=digits))
    print(
        "  (* = 95% CI が 0 をまたがない / n@5%: 5% の差を語るのに要る episodes"
        " / r: score との相関)"
    )

    # 本命の判定は score のペア差。ここだけは結論を文章で書く。
    score_stats = paired_stats(_column(base, "score"), _column(new, "score"))
    print(
        f"\n  score のペア差: {score_stats.delta:+.1f} "
        f"(差の SD={_fmt(score_stats.sd_diff, 1)}, SE={_fmt(score_stats.se, 1)})"
    )
    if score_stats.sd_diff == 0.0:
        print("  全ペアで score が完全一致。検定するまでもなく差は無い。")
    elif score_stats.significant:
        print("  95% CI が 0 をまたがない。この n で有意。")
    else:
        need = score_stats.required_n(100.0)
        print(
            "  95% CI が 0 をまたぐ = この n では有意差なし。"
            f" +/-100 点を語るには n={_fmt(need, 0)} 程度が必要。"
        )

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
