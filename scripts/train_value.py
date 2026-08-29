"""`collect_value.py` のデータに V を当てて、同点帯に順序が入るかを測る。

**これは方策ではなく足切り。** 学習した V で候補を並べ直したとき、教師の
同点帯の外へ手が出るかだけを見る。判定の基準は `band_escape.py` と同じで、
帯の中で手が変わっても score は動かないと n=133 で測ってあるため
(NOTES「決着: 同点帯は本当に無差別」)。ここが数 % なら、V を厚くしても
A/B では何も出ない。

候補の値は Q =（その手の本家点）+ V（落下後の盤）。V のラベルは実リターンで、
教師の eval ではない（→NOTES「決めた: 価値は教師の eval ではなく実リターンから学ぶ」）。

当てるのは ridge（閉形式・numpy だけ）。まず線形で引くのは、ここで帯が割れない
なら特徴が足りないと分かり、モデルを厚くしても同じところで止まるため。
scipy は意図して入れていないので使わない。

用法:
  python scripts/train_value.py artifacts/value_20ep.npz
  python scripts/train_value.py artifacts/value_20ep.npz --eps 0.5 --alpha 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT

from src.training.features import FEATURE_NAMES

# 帯の幅。band_escape の既定と揃える。
DEFAULT_EPS = 0.1
DEFAULT_DATA = ROOT / "artifacts" / "value_dataset.npz"
# held-out に回すエピソードの割合。行ではなくエピソードで割る
# (同じ局の行は強く相関するので、行で割ると held-out が漏れる)。
HOLDOUT_FRAC = 0.25


def _standardize(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z 化のための平均と標準偏差。特徴のスケールは項ごとに 3 桁違う。"""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    # 定数列 (どの盤でも同じ値) は 0 のままにして、割り算で飛ばさない。
    std[std < 1e-8] = 1.0
    return mean, std


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """閉形式の ridge。返すのは bias 込みの係数 (D+1,)。"""
    n, d = x.shape
    design = np.hstack([x, np.ones((n, 1))])
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0  # bias は罰しない
    return np.linalg.solve(design.T @ design + reg, design.T @ y)


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w[:-1] + w[-1]


def _r2(pred: np.ndarray, truth: np.ndarray) -> float:
    resid = float(((truth - pred) ** 2).sum())
    total = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - resid / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        type=Path,
        nargs="?",
        default=DEFAULT_DATA,
        help=f"collect_value.py が吐いた npz (既定 {DEFAULT_DATA.name})",
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="同点帯の幅")
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge の罰則")
    parser.add_argument(
        "--keep-truncated",
        action="store_true",
        help="打ち切りエピソードも学習に使う (既定は落とす。リターンが欠けている)",
    )
    args = parser.parse_args()

    data = np.load(args.data)
    feats = data["feats"]
    returns = data["returns"]
    episodes = data["episodes"]
    steps = data["steps"]
    truncated = data["truncated"]

    keep = np.ones(len(feats), dtype=bool) if args.keep_truncated else ~truncated
    dropped = int((~keep).sum())
    feats, returns, episodes, steps = (
        feats[keep],
        returns[keep],
        episodes[keep],
        steps[keep],
    )
    if len(feats) == 0:
        raise SystemExit("学習に使える手が残らなかった")

    ep_ids = np.array(sorted(set(episodes.tolist())))
    n_test = max(1, int(len(ep_ids) * HOLDOUT_FRAC))
    test_ids = set(ep_ids[-n_test:].tolist())
    is_test = np.array([e in test_ids for e in episodes])

    print(f"データ {args.data}")
    print(
        f"手 {len(feats)}   エピソード {len(ep_ids)}   "
        f"打ち切りで落とした手 {dropped}"
    )
    print(f"held-out: {n_test} エピソード / {int(is_test.sum())} 手\n")

    mean, std = _standardize(feats[~is_test])
    x = (feats - mean) / std
    w = _ridge(x[~is_test], returns[~is_test], args.alpha)

    train_pred = _predict(x[~is_test], w)
    test_pred = _predict(x[is_test], w)
    print("=== 当てはまり (リターン = その盤から先で取った点) ===")
    print(f"  train R^2 {_r2(train_pred, returns[~is_test]):.3f}")
    print(f"  test  R^2 {_r2(test_pred, returns[is_test]):.3f}")
    # V が「あと何手残っているか」を数えているだけだと、同じ局面の候補どうしを
    # 分けられない。手番との相関を並べて出しておく。
    print(f"  r(V, 手番)      {np.corrcoef(_predict(x, w), steps)[0, 1]:+.3f}")
    print(f"  r(リターン, 手番) {np.corrcoef(returns, steps)[0, 1]:+.3f}")

    print("\n=== 係数 (z 化した特徴。符号がその盤の性質の向き) ===")
    order = np.argsort(-np.abs(w[:-1]))
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<18}{w[i]:+9.1f}")

    # --- 足切り: 帯の外へ出るか ---
    cand_feats = data["cand_feats"]
    cand_rewards = data["cand_rewards"]
    cand_evals = data["cand_evals"]
    cand_chosen = data["cand_chosen"]
    cand_row = data["cand_row"]
    if len(cand_row) == 0:
        raise SystemExit("候補表が無い (--candidate-stride 0 で集めた?)")

    cand_v = _predict((cand_feats - mean) / std, w)
    cand_q = cand_rewards + cand_v

    changed = escaped = 0
    positions = 0
    band_sizes: list[int] = []
    v_ranges: list[float] = []
    v_band_ranges: list[float] = []
    flat_in_band = 0
    for row in sorted(set(cand_row.tolist())):
        mask = cand_row == row
        evals = cand_evals[mask]
        if len(evals) < 2:
            continue
        positions += 1
        band = evals >= evals.max() - args.eps
        band_sizes.append(int(band.sum()))

        q = cand_q[mask]
        v = cand_v[mask]
        v_ranges.append(float(v.max() - v.min()))
        v_band = v[band]
        v_band_range = float(v_band.max() - v_band.min())
        v_band_ranges.append(v_band_range)
        if v_band_range < 1e-9:
            flat_in_band += 1

        teacher = int(np.flatnonzero(cand_chosen[mask])[0])
        pick = int(np.argmax(q))
        if pick != teacher:
            changed += 1
        if not band[pick]:
            escaped += 1

    # 候補間で動かない特徴は、重みが何倍でも argmax から消える
    # (→NOTES「今の手で動かせない盤の性質には減点を付けない」)。当てはまりに
    # 効いていても順位には効かないので、両方を並べて出す。
    print("\n=== 候補間で動くか (同じ局面の候補どうしの散らばり) ===")
    print("  特徴                レンジ中央   全候補が同値")
    rows_sorted = sorted(set(cand_row.tolist()))
    movable: list[int] = []
    for i, name in enumerate(FEATURE_NAMES):
        spans = []
        same = 0
        for row in rows_sorted:
            col = cand_feats[cand_row == row, i]
            span = float(col.max() - col.min())
            spans.append(span)
            if span < 1e-9:
                same += 1
        frac = same / len(spans)
        if frac < 0.99:
            movable.append(i)
        print(f"  {name:<18}{statistics.median(spans):11.3f}   {frac * 100:5.1f}%")

    dead = [n for i, n in enumerate(FEATURE_NAMES) if i not in movable]
    if dead:
        print(f"\n  順位に効かない (ほぼ常に全候補同値): {', '.join(dead)}")
    if movable and len(movable) < len(FEATURE_NAMES):
        # 動く特徴だけで引き直す。当てはまりのうち手を選べる分はどれだけか。
        w_mov = _ridge(x[~is_test][:, movable], returns[~is_test], args.alpha)
        r2_mov = _r2(_predict(x[is_test][:, movable], w_mov), returns[is_test])
        print(
            f"  動く特徴だけの test R^2 {r2_mov:.3f}   "
            f"(全部入り {_r2(test_pred, returns[is_test]):.3f})"
        )

    print(f"\n=== 足切り: V で並べ直すと帯の外へ出るか (eps={args.eps}) ===")
    print(f"  局面 {positions}   帯の候補数 中央 {statistics.median(band_sizes):.0f}")
    print(
        f"  手が変わる    {changed:>5}/{positions} ({changed / positions * 100:5.1f}%)"
    )
    print(
        f"  帯の外へ出る  {escaped:>5}/{positions} ({escaped / positions * 100:5.1f}%)"
    )
    print(
        f"  V の候補間レンジ 中央 {statistics.median(v_ranges):.2f}   "
        f"帯内 中央 {statistics.median(v_band_ranges):.2f}"
    )
    print(
        f"  帯内で全候補が同値  {flat_in_band}/{positions} "
        f"({flat_in_band / positions * 100:.1f}%)"
    )
    print(
        "\n帯の外へ出るのが数 % なら、A/B を回しても score は動かない"
        "\n(→NOTES「足切りは『帯の外へ出るか』で測る」)。"
    )


if __name__ == "__main__":
    main()
