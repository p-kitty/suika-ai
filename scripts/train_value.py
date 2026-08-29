"""Fit V on `collect_value.py` data and measure whether it puts an order into the tie band.

**This is a screen, not a policy.** When candidates are reordered by the learned V, it only looks at whether moves
leave the teacher's tie band. The criterion is the same as `band_escape.py`, because
moves changing inside the band were measured not to move score at n=133
(NOTES 'Settled: the tie band really is indifferent'). If this is a few %, thickening V
shows nothing in an A/B.

A candidate's value is Q = (that move's real-game score) + V (post-drop board). V's label is the realized return,
not the teacher's eval (→NOTES 'Decided: learn value from realized returns, not the teacher's eval').

The fit is ridge (closed form, numpy only). Starting linear is because if the band does not split here
the features are shown to be lacking, and a thicker model would stop at the same place.
scipy is deliberately not installed, so it is not used.

Usage:
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

# Band width. Matches the band_escape default.
DEFAULT_EPS = 0.1
DEFAULT_DATA = ROOT / "artifacts" / "value_dataset.npz"
# Fraction of episodes held out. Split by episode, not by row
# (rows of the same game are strongly correlated, so splitting by row leaks into the held-out set).
HOLDOUT_FRAC = 0.25


def _standardize(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation for standardization. Feature scales differ by 3 orders of magnitude per term."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    # Constant columns (same value on every board) stay at 0 instead of blowing up in the division.
    std[std < 1e-8] = 1.0
    return mean, std


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge. Returns coefficients including the bias (D+1,)."""
    n, d = x.shape
    design = np.hstack([x, np.ones((n, 1))])
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0  # the bias is not penalized
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
        help=f"npz written by collect_value.py (default {DEFAULT_DATA.name})",
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="width of the tie band")
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
    parser.add_argument(
        "--keep-truncated",
        action="store_true",
        help="also use truncated episodes for training (dropped by default; their returns are missing)",
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
        raise SystemExit("no moves left usable for training")

    ep_ids = np.array(sorted(set(episodes.tolist())))
    n_test = max(1, int(len(ep_ids) * HOLDOUT_FRAC))
    test_ids = set(ep_ids[-n_test:].tolist())
    is_test = np.array([e in test_ids for e in episodes])

    print(f"data {args.data}")
    print(
        f"moves {len(feats)}   episodes {len(ep_ids)}   "
        f"moves dropped for truncation {dropped}"
    )
    print(f"held-out: {n_test} episodes / {int(is_test.sum())} moves\n")

    mean, std = _standardize(feats[~is_test])
    x = (feats - mean) / std
    w = _ridge(x[~is_test], returns[~is_test], args.alpha)

    train_pred = _predict(x[~is_test], w)
    test_pred = _predict(x[is_test], w)
    print("=== fit (return = points scored after that board) ===")
    print(f"  train R^2 {_r2(train_pred, returns[~is_test]):.3f}")
    print(f"  test  R^2 {_r2(test_pred, returns[is_test]):.3f}")
    # If V only counts 'how many moves are left', it cannot separate candidates of the same position.
    # Show the correlation with the move number alongside.
    print(f"  r(V, move number)      {np.corrcoef(_predict(x, w), steps)[0, 1]:+.3f}")
    print(f"  r(return, move number) {np.corrcoef(returns, steps)[0, 1]:+.3f}")

    print("\n=== coefficients (standardized features; the sign is the direction of that board property) ===")
    order = np.argsort(-np.abs(w[:-1]))
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<18}{w[i]:+9.1f}")

    # --- Screen: does it escape the band ---
    cand_feats = data["cand_feats"]
    cand_rewards = data["cand_rewards"]
    cand_evals = data["cand_evals"]
    cand_chosen = data["cand_chosen"]
    cand_row = data["cand_row"]
    if len(cand_row) == 0:
        raise SystemExit("no candidate table (collected with --candidate-stride 0?)")

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

    # Features that do not move between candidates vanish from argmax however large the weight
    # (→NOTES 'Do not penalize board properties the current move cannot change'). Even if they help the fit
    # they do not affect ranking, so both are shown side by side.
    print("\n=== does it move between candidates (spread among candidates of the same position) ===")
    print("  feature             median range   all candidates equal")
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
        print(f"\n  no effect on ranking (almost always equal for all candidates): {', '.join(dead)}")
    if movable and len(movable) < len(FEATURE_NAMES):
        # Refit with only moving features. How much of the fit can choose moves.
        w_mov = _ridge(x[~is_test][:, movable], returns[~is_test], args.alpha)
        r2_mov = _r2(_predict(x[is_test][:, movable], w_mov), returns[is_test])
        print(
            f"  test R^2 with moving features only {r2_mov:.3f}   "
            f"(all features {_r2(test_pred, returns[is_test]):.3f})"
        )

    print(f"\n=== screen: does reordering by V escape the band (eps={args.eps}) ===")
    print(f"  positions {positions}   band candidate count median {statistics.median(band_sizes):.0f}")
    print(
        f"  moves change      {changed:>5}/{positions} ({changed / positions * 100:5.1f}%)"
    )
    print(
        f"  escapes the band  {escaped:>5}/{positions} ({escaped / positions * 100:5.1f}%)"
    )
    print(
        f"  V candidate range median {statistics.median(v_ranges):.2f}   "
        f"inside the band median {statistics.median(v_band_ranges):.2f}"
    )
    print(
        f"  all candidates equal inside the band  {flat_in_band}/{positions} "
        f"({flat_in_band / positions * 100:.1f}%)"
    )
    print(
        "\nIf only a few % escape the band, running an A/B will not move score"
        "\n(→NOTES 'Screen on does it escape the band')."
    )


if __name__ == "__main__":
    main()
