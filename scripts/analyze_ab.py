"""compare_policy --out のダンプを読み直し、代理指標を選ぶための道具。

score はノイズが大きく、±100 点を語るのに n≈100 要る。1 回の A/B に数時間
かかるので、これが方策改善の律速になっている。そこで「score より少ない
エピソード数で同じ変化を捉えられる指標」を探す。

各指標について、同一シードのペア差から次を出す:

- n_detect: 今回観測された差を 95% CI で 0 から離すのに要るエピソード数。
  **これが小さい指標ほど、同じ変化を早く検出できる = 代理指標に向く。**
- 感度比: |t| / |score の t|。1 より大きければ score より鋭い。
- ペア化利得: シードを対にしたことで差の標準誤差が何割縮んだか。独立な
  シードで測ると差の SD は個体差の sqrt(2) 倍になるので、そこを基準に取る。
  0 なら対にした意味が無く、0.5 ならペア比較で SE が半分。
- r: score との相関。これが 0 に近い指標は、速く測れても別のものを測っている。

**選定バイアスに注意。** 同じデータで一番良く見えた指標を選ぶと、たまたま
当たりを引いただけのものを選ぶ。ここで上位に来た指標は、別の変更・別の
シード列で撮った 2 本目のダンプでも上位かを確認してから採用すること。
複数のダンプを引数に並べれば、指標ごとに全ダンプぶんの行が出る。

用法:
  python scripts/analyze_ab.py artifacts/packed_ab_n100.json
  python scripts/analyze_ab.py artifacts/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import correlation, detect_n, paired_stats, pairing_gain

# seed は指標ではない。
SKIP = {"seed"}


def _fmt(value: float, digits: int = 2, width: int = 8) -> str:
    return f"{'-':>{width}}" if value != value else f"{value:>{width}.{digits}f}"


def _report(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    a_rows: list[dict[str, float]] = data["a"]
    b_rows: list[dict[str, float]] = data["b"]
    keys = [k for k in a_rows[0] if k not in SKIP]

    scores_a = [r["score"] for r in a_rows]
    scores_b = [r["score"] for r in b_rows]
    t_score = abs(paired_stats(scores_a, scores_b).t)

    print(
        f"\n{path.name}: {data.get('variant', '?')} "
        f"episodes={data.get('episodes', len(a_rows))} "
        f"max_steps={data.get('max_steps', '?')} seed={data.get('seed', '?')}"
    )
    capped = sum(
        1 for r in a_rows + b_rows if r["steps"] >= float(data.get("max_steps", 0))
    )
    if capped:
        print(f"  注意: {capped}/{len(a_rows) + len(b_rows)} が max_steps で打ち切り")
    print(
        "  指標           delta   sd_diff  n_detect   感度比  ペア化利得   r(score)"
    )
    rows = []
    for key in keys:
        stats = paired_stats([r[key] for r in a_rows], [r[key] for r in b_rows])
        gain = pairing_gain(stats.sd_diff, stats.sd_pooled)
        rows.append(
            (
                detect_n(stats.delta, stats.sd_diff),
                key,
                stats,
                gain,
                correlation(
                    [r[key] for r in a_rows] + [r[key] for r in b_rows],
                    scores_a + scores_b,
                ),
            )
        )
    # n_detect の小さい順 = 早く決着がつく指標順。NaN（差が出ていない）は末尾へ。
    rows.sort(key=lambda r: (r[0] != r[0], r[0]))
    for detect, key, stats, gain, r in rows:
        sens = abs(stats.t) / t_score if t_score else float("nan")
        print(
            f"  {key:<12}{_fmt(stats.delta, 2, 9)} {_fmt(stats.sd_diff, 2)}"
            f" {_fmt(detect, 0, 9)} {_fmt(sens, 2)} {_fmt(gain, 2, 11)}"
            f" {_fmt(r, 2, 10)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.dumps:
        _report(path)
    if len(args.dumps) > 1:
        print(
            "\n  複数ダンプで揃って上位に来た指標だけを代理指標として採用すること。"
        )


if __name__ == "__main__":
    main()
