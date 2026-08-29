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


def _step_trend(
    steps: np.ndarray, returns: np.ndarray, *, window: int = 11
) -> np.ndarray:
    """Mean return per move number. Returns a lookup table of move number -> mean.

    Regardless of how good the board is, the return is mostly decided by 'how many moves are left'
    (measured r=-0.815). Subtracting that before fitting leaves 'is the board good
    compared with the average at the same move number'. The late game has few samples and is noisy, so it is smoothed with a moving average.

    **The ranking itself does not change** (candidates of the same position share the move number, so the amount subtracted is
    the same). What changes is how the coefficients land: they are no longer absorbed by terms that explain the clock.
    """
    top = int(steps.max())
    means = np.full(top + 1, np.nan)
    for s in range(top + 1):
        mask = steps == s
        if mask.any():
            means[s] = returns[mask].mean()
    # Fill gaps with neighboring values before smoothing.
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


# Horizons (None = to the end) and penalties to sweep.
SWEEP_HORIZONS: tuple[int | None, ...] = (None, 100, 30, 10, 3, 1)
SWEEP_ALPHAS: tuple[float, ...] = (1.0, 100.0, 10000.0)
SWEEP_FOLDS = 4


def _horizon_return(
    rewards: np.ndarray,
    episodes: np.ndarray,
    ep_ids: np.ndarray,
    horizon: int | None,
) -> np.ndarray:
    """Points over the next horizon moves from that board. To the end of the game if None."""
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
    """Cross-validated R^2 over horizon × alpha. Shows at which time scale the signal is.

    A single held-out split cannot be read, so episodes are split into `SWEEP_FOLDS` folds.
    The detrend side is the real one; if it is 0, 'how good the board is' is not being predicted
    (the high R^2 without detrend is earned only from move number information).
    """
    print(f"=== cross-validated R^2 over horizon × alpha ({SWEEP_FOLDS}-fold, split by episode) ===")
    print(f"{'label':>10} {'alpha':>8} {'no detrend':>18} {'detrend':>18}")
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
            name = "to the end" if horizon is None else f"{horizon} moves"
            print(
                f"{name:>10} {alpha:>8.0f}"
                f" {cells[0][0]:>11.3f}±{cells[0][1]:.3f}"
                f" {cells[1][0]:>11.3f}±{cells[1][1]:.3f}"
            )
    print(
        "\nIf the detrend side is 0, the fit is only move number information, and how good the board is"
        "\nis not predicted. Rows whose between-fold SD exceeds the mean cannot be read."
    )


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
    parser.add_argument(
        "--detrend",
        action="store_true",
        help="subtract the per-move-number mean from the label (stops V from learning the clock)",
    )
    parser.add_argument(
        "--drop-dead",
        action="store_true",
        help="fit without features that barely move between candidates (they never affect ranking anyway)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="only run cross-validation over horizon × alpha and exit (where the signal is)",
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

    if args.sweep:
        _sweep(feats, data["rewards"][keep], episodes, steps, ep_ids)
        return

    cand_feats = data["cand_feats"]
    cand_rewards = data["cand_rewards"]
    cand_evals = data["cand_evals"]
    cand_chosen = data["cand_chosen"]
    cand_row = data["cand_row"]
    if len(cand_row) == 0:
        raise SystemExit("no candidate table (collected with --candidate-stride 0?)")

    # Report first whether features move between candidates. Used for the --drop-dead selection and the table below.
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
        raise SystemExit("no usable features left")

    label = returns
    trend = None
    if args.detrend:
        trend = _step_trend(steps[~is_test], returns[~is_test])
        label = returns - trend[np.clip(steps, 0, len(trend) - 1)]

    mean, std = _standardize(feats[~is_test])
    x = ((feats - mean) / std)[:, use]
    w = _ridge(x[~is_test], label[~is_test], args.alpha)

    train_pred = _predict(x[~is_test], w)
    test_pred = _predict(x[is_test], w)
    what = "return − mean per move number" if args.detrend else "return"
    print(f"=== fit (label = {what}) ===")
    print(f"  features {len(use)}/{len(FEATURE_NAMES)}" + ("  (only those that move)" if args.drop_dead else ""))
    print(f"  train R^2 {_r2(train_pred, label[~is_test]):.3f}")
    print(f"  test  R^2 {_r2(test_pred, label[is_test]):.3f}")
    # If V only counts 'how many moves are left', it cannot separate candidates of the same position.
    # Show the correlation with the move number alongside.
    print(f"  r(V, move number)      {np.corrcoef(_predict(x, w), steps)[0, 1]:+.3f}")
    print(f"  r(label, move number) {np.corrcoef(label, steps)[0, 1]:+.3f}")

    print("\n=== coefficients (standardized features; the sign is the direction of that board property) ===")
    order = np.argsort(-np.abs(w[:-1]))
    for i in order:
        print(f"  {FEATURE_NAMES[use[i]]:<18}{w[i]:+9.1f}")

    # --- Screen: does it escape the band ---
    # Detrending does not change the ranking (the amount subtracted is the same within a position), so here
    # rank directly by Q = real-game score + V.
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

    # Features that do not move between candidates vanish from argmax however large the weight
    # (→NOTES 'Do not penalize board properties the current move cannot change'). Even if they help the fit
    # they do not affect ranking, so both are shown side by side.
    print("\n=== does it move between candidates (spread among candidates of the same position) ===")
    print("  feature             median range   all candidates equal")
    for i, name in enumerate(FEATURE_NAMES):
        print(
            f"  {name:<18}{statistics.median(spans_by_feature[i]):11.3f}"
            f"   {same_frac[i] * 100:5.1f}%"
        )
    dead = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if i not in movable]
    if dead:
        print(f"\n  no effect on ranking (almost always equal for all candidates): {', '.join(dead)}")

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
