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

**足切りに R² を使わない。** R² は目盛りのずれも罰するので、分散の大きい
最終リターンでは順位が当たっていても 0 に潰れる（実測: R² 0.012 に対し r 0.286）。
順位しか使わないので見るのは相関のほう。

用法:
  python scripts/train_value.py --sweep                    # 信号がどの地平線にあるか
  python scripts/train_value.py --horizon 100 --detrend --drop-dead
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


def _step_trend(
    steps: np.ndarray, returns: np.ndarray, *, window: int = 11
) -> np.ndarray:
    """手番ごとの平均リターン。手番 -> 平均 の引き表を返す。

    盤の良し悪しと関係なく、リターンは「あと何手残っているか」でほぼ決まる
    (実測 r=-0.815)。その分を引いてから当てると、残るのは「同じ手番の平均と
    比べて良い盤か」になる。終盤は本数が減って荒れるので移動平均で均す。

    **順位そのものは変わらない**（同じ局面の候補は手番が同じなので、引く量も
    同じ）。変わるのは係数の当たり方で、時計を説明する項に吸われなくなる。
    """
    top = int(steps.max())
    means = np.full(top + 1, np.nan)
    for s in range(top + 1):
        mask = steps == s
        if mask.any():
            means[s] = returns[mask].mean()
    # 空きは前後の値で埋めてから均す。
    idx = np.arange(top + 1)
    good = ~np.isnan(means)
    means = np.interp(idx, idx[good], means[good])
    pad = window // 2
    padded = np.pad(means, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def _r2(pred: np.ndarray, truth: np.ndarray) -> float:
    resid = float(((truth - pred) ** 2).sum())
    total = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - resid / total if total > 0 else 0.0


# 掃引する地平線 (None = 最後まで) と罰則。
SWEEP_HORIZONS: tuple[int | None, ...] = (None, 100, 30, 10, 3, 1)
SWEEP_ALPHAS: tuple[float, ...] = (1.0, 100.0, 10000.0)
SWEEP_FOLDS = 4


def _horizon_return(
    rewards: np.ndarray,
    episodes: np.ndarray,
    ep_ids: np.ndarray,
    horizon: int | None,
) -> np.ndarray:
    """その盤から先 horizon 手ぶんの点。None なら局の最後まで。"""
    out = np.zeros(len(rewards), dtype=np.float64)
    for ep in ep_ids:
        idx = np.flatnonzero(episodes == ep)
        r = rewards[idx]
        cum = np.concatenate([[0.0], np.cumsum(r)])
        for j in range(len(idx)):
            hi = len(r) if horizon is None else min(j + 1 + horizon, len(r))
            out[idx[j]] = cum[hi] - cum[j + 1]
    return out


def _sweep(
    feats: np.ndarray,
    rewards: np.ndarray,
    episodes: np.ndarray,
    steps: np.ndarray,
    ep_ids: np.ndarray,
) -> None:
    """地平線 × alpha の交差検証 R^2。信号がどの時間尺度にあるかを見る。

    held-out 1 本では読めないので、エピソードを `SWEEP_FOLDS` 分割して回す。
    detrend 側が本命で、そちらが 0 なら「盤の良し悪し」は当たっていない
    （detrend 無しの高い R^2 は手番の情報でしか稼げていない）。
    """
    print(f"=== 地平線 × alpha の交差検証 R^2 ({SWEEP_FOLDS}-fold, エピソード分割) ===")
    print(f"{'ラベル':>10} {'alpha':>8} {'detrend なし':>18} {'detrend あり':>18}")
    groups = np.array_split(ep_ids, SWEEP_FOLDS)
    for horizon in SWEEP_HORIZONS:
        y = _horizon_return(rewards, episodes, ep_ids, horizon)
        for alpha in SWEEP_ALPHAS:
            cells = []
            for detrend in (False, True):
                scores = []
                for group in groups:
                    test = np.isin(episodes, group)
                    if test.all() or not test.any():
                        continue
                    label = y
                    if detrend:
                        trend = _step_trend(steps[~test], y[~test])
                        label = y - trend[np.clip(steps, 0, len(trend) - 1)]
                    mean, std = _standardize(feats[~test])
                    x = (feats - mean) / std
                    w = _ridge(x[~test], label[~test], alpha)
                    scores.append(_r2(_predict(x[test], w), label[test]))
                cells.append((float(np.mean(scores)), float(np.std(scores))))
            name = "最後まで" if horizon is None else f"{horizon} 手"
            print(
                f"{name:>10} {alpha:>8.0f}"
                f" {cells[0][0]:>11.3f}±{cells[0][1]:.3f}"
                f" {cells[1][0]:>11.3f}±{cells[1][1]:.3f}"
            )
    print(
        "\ndetrend 側が 0 なら、当てはまりは手番の情報だけで、盤の良し悪しは"
        "\n当たっていない。fold 間の標準偏差が平均を超えている行は読めていない。"
    )


# 局どうしを比べる手番。序盤すぎると盤が育っておらず、遅すぎると届く局が減る。
CARRY_PROBES = (40, 60, 100)


def _carry(
    feats: np.ndarray,
    per_move: np.ndarray,
    score_per_move: np.ndarray,
    episodes: np.ndarray,
    steps: np.ndarray,
    ep_ids: np.ndarray,
    alpha: float,
) -> None:
    """V が「局として高い score で終わるか」を当てられるかを見る。

    **手番を固定して局どうしを比べる。** 局面をまたいでプールすると、同じ局の
    連続する盤は V も残り点も似ているので相関が水増しされる（実測: プールで
    r=0.28、手番固定で r=0.09）。的はラベルが何であれ本家点。
    """
    full_score = _horizon_return(score_per_move, episodes, ep_ids, None)
    groups = np.array_split(ep_ids, SWEEP_FOLDS)
    print("=== 局間に伝わるか (手番を固定して局どうしを比べる) ===")
    print(f"{'当てた地平線':>12} {'手番':>6}   r(V, その局の残り点)")
    for horizon in (100, 30, None):
        y = _horizon_return(per_move, episodes, ep_ids, horizon)
        for probe in CARRY_PROBES:
            rs: list[float] = []
            counts: list[int] = []
            for group in groups:
                test = np.isin(episodes, group)
                trend = _step_trend(steps[~test], y[~test])
                label = y - trend[np.clip(steps, 0, len(trend) - 1)]
                mean, std = _standardize(feats[~test])
                x = (feats - mean) / std
                w = _ridge(x[~test], label[~test], alpha)
                sel = test & (steps == probe)
                if int(sel.sum()) < 5:
                    continue
                rs.append(float(np.corrcoef(_predict(x[sel], w), full_score[sel])[0, 1]))
                counts.append(int(sel.sum()))
            if not rs:
                continue
            name = "最後まで" if horizon is None else f"{horizon} 手"
            print(
                f"{name:>12} {probe:>6}   {np.mean(rs):+.3f} ± {np.std(rs):.3f}"
                f"   (局 {int(np.mean(counts))}/fold)"
            )
    print(
        "\n±SD が平均を覆っている行は 0 と区別が付かない。NOTES が逆転率について"
        "\n記録した r=0.00〜0.12 と同じ range なら、エピソード単位では使えない。"
    )


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
    parser.add_argument(
        "--detrend",
        action="store_true",
        help="ラベルから手番の平均を引く (V が時計を覚えるのを止める)",
    )
    parser.add_argument(
        "--drop-dead",
        action="store_true",
        help="候補間でほぼ動かない特徴を外して当てる (順位には元から効かない)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="地平線 × alpha の交差検証だけ回して終わる (信号がどこにあるか)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        metavar="K",
        help="ラベルを先 K 手ぶんの点にする (既定は局の最後まで)",
    )
    parser.add_argument(
        "--label",
        choices=("score", "cascades"),
        default="score",
        help="数える量。cascades は 1 手 3 合成以上の回数 (score より低分散)",
    )
    parser.add_argument(
        "--carry",
        action="store_true",
        help="局間に伝わるかだけ測る (手番を固定して局どうしを比べる)",
    )
    args = parser.parse_args()

    data = np.load(args.data)
    feats = data["feats"]
    episodes = data["episodes"]
    steps = data["steps"]
    truncated = data["truncated"]

    if args.label == "cascades" and "merges" not in data:
        raise SystemExit(
            f"{args.data} に merges が無い。cascades を数えるには集め直しが要る"
        )
    # 手ごとに数える量。cascades は 1 手 3 合成以上 (NOTES で score に対して
    # 検証済みの唯一の代理指標。感度比 1.34〜1.40、r(score) 0.83)。
    per_move = (
        data["rewards"].astype(np.float64)
        if args.label == "score"
        else (data["merges"] >= 3).astype(np.float64)
    )

    keep = np.ones(len(feats), dtype=bool) if args.keep_truncated else ~truncated
    dropped = int((~keep).sum())
    feats, episodes, steps, per_move = (
        feats[keep],
        episodes[keep],
        steps[keep],
        per_move[keep],
    )
    if len(feats) == 0:
        raise SystemExit("学習に使える手が残らなかった")
    # 最終 score へ伝わるかを見る側の的は、ラベルが何であれ本家点。
    score_per_move = data["rewards"].astype(np.float64)[keep]

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

    print(f"数える量: {args.label}\n")
    if args.sweep:
        _sweep(feats, per_move, episodes, steps, ep_ids)
        return
    if args.carry:
        _carry(feats, per_move, score_per_move, episodes, steps, ep_ids, args.alpha)
        return

    returns = _horizon_return(per_move, episodes, ep_ids, None)

    cand_feats = data["cand_feats"]
    cand_rewards = data["cand_rewards"]
    cand_evals = data["cand_evals"]
    cand_chosen = data["cand_chosen"]
    cand_row = data["cand_row"]
    if len(cand_row) == 0:
        raise SystemExit("候補表が無い (--candidate-stride 0 で集めた?)")

    # 候補間で動くかを先に出す。--drop-dead の選別にも、下の表にも使う。
    rows_sorted = sorted(set(cand_row.tolist()))
    spans_by_feature: list[list[float]] = []
    same_frac: list[float] = []
    for i in range(len(FEATURE_NAMES)):
        spans = []
        same = 0
        for row in rows_sorted:
            col = cand_feats[cand_row == row, i]
            span = float(col.max() - col.min())
            spans.append(span)
            if span < 1e-9:
                same += 1
        spans_by_feature.append(spans)
        same_frac.append(same / len(spans))
    movable = [i for i, frac in enumerate(same_frac) if frac < 0.99]

    use = movable if args.drop_dead else list(range(len(FEATURE_NAMES)))
    if not use:
        raise SystemExit("使える特徴が残らなかった")

    if args.horizon is None:
        label = returns
    else:
        label = _horizon_return(per_move, episodes, ep_ids, args.horizon)
    if args.detrend:
        trend = _step_trend(steps[~is_test], label[~is_test])
        label = label - trend[np.clip(steps, 0, len(trend) - 1)]

    mean, std = _standardize(feats[~is_test])
    x = ((feats - mean) / std)[:, use]
    w = _ridge(x[~is_test], label[~is_test], args.alpha)

    train_pred = _predict(x[~is_test], w)
    test_pred = _predict(x[is_test], w)
    what = "最後まで" if args.horizon is None else f"先 {args.horizon} 手"
    if args.detrend:
        what += " − 手番ごとの平均"
    print(f"=== 当てはまり (ラベル = {what}) ===")
    print(f"  特徴 {len(use)}/{len(FEATURE_NAMES)} 本" + ("  (動く分だけ)" if args.drop_dead else ""))
    print(f"  train R^2 {_r2(train_pred, label[~is_test]):.3f}")
    print(f"  test  R^2 {_r2(test_pred, label[is_test]):.3f}")
    # V が「あと何手残っているか」を数えているだけだと、同じ局面の候補どうしを
    # 分けられない。手番との相関を並べて出しておく。
    print(f"  r(V, 手番)      {np.corrcoef(_predict(x, w), steps)[0, 1]:+.3f}")
    print(f"  r(ラベル, 手番) {np.corrcoef(label, steps)[0, 1]:+.3f}")

    print("\n=== 係数 (z 化した特徴。符号がその盤の性質の向き) ===")
    order = np.argsort(-np.abs(w[:-1]))
    for i in order:
        print(f"  {FEATURE_NAMES[use[i]]:<18}{w[i]:+9.1f}")

    # --- 足切り: 帯の外へ出るか ---
    # detrend しても順位は変わらない（引く量は局面内で同じ）ので、ここは
    # そのまま Q = 本家点 + V で並べる。
    cand_v = _predict(((cand_feats - mean) / std)[:, use], w)
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
    for i, name in enumerate(FEATURE_NAMES):
        print(
            f"  {name:<18}{statistics.median(spans_by_feature[i]):11.3f}"
            f"   {same_frac[i] * 100:5.1f}%"
        )
    dead = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if i not in movable]
    if dead:
        print(f"\n  順位に効かない (ほぼ常に全候補同値): {', '.join(dead)}")

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
