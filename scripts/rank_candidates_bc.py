"""Can a student rank candidates from per-candidate post-drop features?

NOTES 'BC does not reach 60-70% match': a 1-layer MLP over the static board plateaued at 29%, and the
stated way out is "per-candidate landing results in the features". Those rows already exist in
artifacts/value_100ep.npz (cand_feats / cand_evals / cand_chosen / cand_row).

Usage:
  python scripts/rank_candidates_bc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT
from src.training.features import FEATURE_NAMES
d = np.load(ROOT / "artifacts" / "value_100ep.npz", allow_pickle=True)
X, ev, chosen, row = d["cand_feats"], d["cand_evals"], d["cand_chosen"], d["cand_row"]
print(f"{len(X)} candidate rows, {len(np.unique(row))} positions, {X.shape[1]} features")

# Group by position.
order = np.argsort(row, kind="stable")
X, ev, chosen, row = X[order], ev[order], chosen[order], row[order]
bounds = np.flatnonzero(np.diff(row)) + 1
groups = np.split(np.arange(len(row)), bounds)
groups = [g for g in groups if len(g) > 1 and chosen[g].sum() == 1]
print(f"usable positions: {len(groups)}")

rng = np.random.default_rng(0)
idx = rng.permutation(len(groups))
cut = int(len(groups) * 0.8)
train, test = [groups[i] for i in idx[:cut]], [groups[i] for i in idx[cut:]]

mu, sd = X.std(0), None
mean = X.mean(0)
sd = np.where(X.std(0) > 1e-9, X.std(0), 1.0)
Z = (X - mean) / sd

def fit_ridge(gs, lam=1.0):
    """Ridge on (candidate - group mean) -> teacher eval, centred per position.

    Centring per position removes the board-level offset, so the fit is about ordering inside a position.
    """
    rows = np.concatenate(gs)
    A = Z[rows].copy()
    y = ev[rows].astype(float).copy()
    start = 0
    for g in gs:
        n = len(g)
        A[start:start + n] -= A[start:start + n].mean(0)
        y[start:start + n] -= y[start:start + n].mean()
        start += n
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return w

def report(w, gs, label, eps=0.1):
    top1 = band = ctrl_top1 = ctrl_band = 0
    for g in gs:
        s = Z[g] @ w
        pick = g[int(np.argmax(s))]
        best = ev[g].max()
        in_band = ev[g] >= best - eps
        top1 += bool(chosen[pick])
        band += bool(in_band[int(np.argmax(s))])
        r = int(rng.integers(len(g)))
        ctrl_top1 += bool(chosen[g[r]])
        ctrl_band += bool(in_band[r])
    n = len(gs)
    print(f"  {label:<10} top1 {top1/n:6.1%} (random {ctrl_top1/n:5.1%})   "
          f"in teacher band {band/n:6.1%} (random {ctrl_band/n:5.1%})")

for lam in (0.1, 1.0, 10.0, 100.0):
    w = fit_ridge(train, lam)
    print(f"lam={lam}")
    report(w, test, "test")

w = fit_ridge(train, 1.0)
print("\ntop weights (|w|):")
for i in np.argsort(-np.abs(w))[:10]:
    print(f"  {FEATURE_NAMES[i]:<20} {w[i]:+.3f}")

# Early/late split: NOTES says the static-board student collapses to 1.4x control late,
# "in the stretch where the board is decided". That is the comparison that matters.
steps_per_move = d["steps"]
print("\nby phase (move number of the position):")
for label, lo, hi in (("early <60", 0, 60), ("mid 60-150", 60, 150), ("late >=150", 150, 10**9)):
    gs = [g for g in test if lo <= steps_per_move[int(row[g[0]])] < hi]
    if gs:
        report(w, gs, label)
