"""B 側だけ回して、保存済みのベースライン (A 側) と突き合わせる。

A 側は変種をまたいで同じ方策なので、A/B のたびに引き直すのは丸ごと無駄。
同じ seed 列で B だけ回し、`--baseline` の JSON にある A 側と対にする。
統計は `compare_policy` と同じペア検定のままで、計算量が半分になる。

ペアリング自体の効きは小さい (実測 r=+0.11〜+0.18、SE の得は 6〜10%)。
手を 1 つ変えると盤が分岐するので、同一 seed でも A と B はほぼ無相関。
それでも対にしておくのは、seed 集合のツモ運を両側で揃えるため。

**ベースラインは方策を変えたら無効になる。** JSON の commit と今の HEAD が
違うときは警告を出すので、無視して使わないこと。

変更は `compare_policy.py` の `_apply_variant` に差す。B 側だけ呼ばれる。

用法:
  python scripts/compare_policy.py --episodes 50 --out artifacts/base.json
  python scripts/compare_b_only.py --baseline artifacts/base.json --episodes 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_policy import _apply_variant, _episode, _head, _line

METRICS = (
    ("score", 2),
    ("early_score", 2),
    ("steps", 1),
    ("merges", 1),
    ("cascades", 2),
    ("max_type", 2),
    ("early_crown", 1),
    ("dead", 3),
    ("dead_early", 3),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.baseline.read_text(encoding="utf-8"))
    base_all = data["a"]
    max_steps = data["max_steps"]
    if args.episodes > len(base_all):
        raise SystemExit(
            f"ベースラインは {len(base_all)} エピソードしかない: --episodes を下げること"
        )
    base = sorted(base_all, key=lambda r: r["seed"])[: args.episodes]
    seeds = [int(r["seed"]) for r in base]

    stamp = data.get("baseline_commit")
    if stamp and stamp != _head():
        print(f"  警告: ベースラインの commit {stamp[:8]} が今の HEAD と違う", flush=True)

    started = time.monotonic()
    print(f"  B (new): {len(seeds)} エピソード実行中... (A 側は再利用)", flush=True)
    rows: list[dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_episode, s, max_steps, True) for s in seeds]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"    {done}/{len(seeds)}  ({time.monotonic() - started:.0f}s)",
                  end="\r", flush=True)
    rows.sort(key=lambda r: r["seed"])
    print(f"  B (new): 完了 {time.monotonic() - started:.0f}s" + " " * 20, flush=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "baseline": str(args.baseline),
                    "episodes": args.episodes,
                    "max_steps": max_steps,
                    "variant": (_apply_variant.__doc__ or "").strip().splitlines()[:1],
                    "a": base,
                    "b": rows,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"  生データを保存: {args.out}")

    print(f"\nepisodes={args.episodes} max_steps={max_steps} "
          f"workers={args.workers} baseline={args.baseline.name}")
    print(f"  A = 保存済みベースライン ({len(base_all)} 本のうち先頭 {args.episodes})")
    for key, digits in METRICS:
        print(_line(key, base, rows, digits=digits))
    print("  (* = 95% CI が 0 をまたがない)")

    from src.util.stats import paired_stats

    stats = paired_stats([r["score"] for r in base], [r["score"] for r in rows])
    print(f"\n  score のペア差: {stats.delta:+.1f} "
          f"(差の SD={stats.sd_diff:.1f}, SE={stats.se:.1f})")
    wins = sum(1 for x, y in zip(base, rows) if y["score"] > x["score"])
    losses = sum(1 for x, y in zip(base, rows) if y["score"] < x["score"])
    print(f"  seed 対戦 (B 視点)  win={wins}  loss={losses}  "
          f"tie={len(base) - wins - losses}")


if __name__ == "__main__":
    main()
