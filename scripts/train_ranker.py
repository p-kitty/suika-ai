"""Fit the candidate ranker on which candidate the teacher chose.

A softmax over the candidates of one position (conditional logit). The ridge fit it replaces targets the
FIRST-ply eval, but the move the teacher plays comes out of the two-ply search, so the labels and the
target disagree. numpy only; the linear fit takes seconds.

Writes artifacts/ranker_logit_h<hidden>.npz per head size. Play them with
`eval_ranker.py --weights`; NOTES 'What the student is missing is the second ply, not fidelity' has
the numbers.

Usage:
  python scripts/train_ranker.py
  python scripts/train_ranker.py artifacts/value_0916_n100.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT

DEFAULT_DATA = ROOT / "artifacts" / "value_0916_n100.npz"
data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
print(f"dataset: {data_path.name}")
d = np.load(data_path, allow_pickle=True)
X, ev, chosen, row = d["cand_feats"], d["cand_evals"], d["cand_chosen"], d["cand_row"]
order = np.argsort(row, kind="stable")
X, ev, chosen, row = X[order], ev[order], chosen[order], row[order]
mean, sd = X.mean(0), np.where(X.std(0) > 1e-9, X.std(0), 1.0)
Z = ((X - mean) / sd).astype(np.float64)

groups = [g for g in np.split(np.arange(len(row)), np.flatnonzero(np.diff(row)) + 1)
          if len(g) > 1 and chosen[g].sum() == 1]
rng = np.random.default_rng(0)
idx = rng.permutation(len(groups))
cut = int(len(groups) * 0.8)
train = [groups[i] for i in idx[:cut]]
test = [groups[i] for i in idx[cut:]]
print(f"{len(groups)} positions  train {len(train)} test {len(test)}  dim {Z.shape[1]}")

# Pack groups into a padded matrix so the softmax is one vectorised pass.
def pack(gs):
    K = max(len(g) for g in gs)
    ids = np.zeros((len(gs), K), dtype=np.int64)
    mask = np.zeros((len(gs), K), dtype=bool)
    tgt = np.zeros(len(gs), dtype=np.int64)
    for i, g in enumerate(gs):
        ids[i, :len(g)] = g
        mask[i, :len(g)] = True
        tgt[i] = int(np.argmax(chosen[g]))
    return ids, mask, tgt

def evaluate(score_fn, gs, label, eps=0.1):
    top1 = band = ctrl = 0
    for g in gs:
        s = score_fn(Z[g])
        p = int(np.argmax(s))
        best = ev[g].max()
        in_band = ev[g] >= best - eps
        top1 += bool(chosen[g][p])
        band += bool(in_band[p])
        ctrl += bool(in_band[int(rng.integers(len(g)))])
    n = len(gs)
    print(f"  {label:<22} top1 {top1/n:6.1%}   band {band/n:6.1%}  (random {ctrl/n:5.1%})")

tr_ids, tr_mask, tr_tgt = pack(train)

def softmax_fit(hidden=0, epochs=300, lr=0.05, l2=1e-4, seed=0):
    r = np.random.default_rng(seed)
    D = Z.shape[1]
    if hidden:
        W1 = r.normal(0, 0.5 / np.sqrt(D), (D, hidden)); b1 = np.zeros(hidden)
        W2 = r.normal(0, 0.5 / np.sqrt(hidden), hidden)
        params = [W1, b1, W2]
    else:
        W1 = np.zeros((D, 1)); params = [W1]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    F = Z[tr_ids]                      # (P, K, D)
    M = tr_mask
    rows_ar = np.arange(len(tr_tgt))
    for ep in range(epochs):
        if hidden:
            H = np.tanh(F @ params[0] + params[1])
            S = H @ params[2]
        else:
            S = (F @ params[0])[:, :, 0]
        S = np.where(M, S, -1e9)
        S -= S.max(1, keepdims=True)
        P = np.exp(S) * M
        P /= P.sum(1, keepdims=True)
        G = P.copy()
        G[rows_ar, tr_tgt] -= 1.0
        G /= len(tr_tgt)
        if hidden:
            gW2 = np.einsum("pk,pkh->h", G, H) + l2 * params[2]
            dH = G[:, :, None] * params[2][None, None, :] * (1 - H ** 2)
            gW1 = np.einsum("pkd,pkh->dh", F, dH) + l2 * params[0]
            gb1 = dH.sum((0, 1))
            grads = [gW1, gb1, gW2]
        else:
            grads = [np.einsum("pk,pkd->d", G, F)[:, None] + l2 * params[0]]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            mh = m[i] / (1 - 0.9 ** (ep + 1)); vh = v[i] / (1 - 0.999 ** (ep + 1))
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
    if hidden:
        return lambda z: np.tanh(z @ params[0] + params[1]) @ params[2], params
    return lambda z: (z @ params[0])[:, 0], params

# Baseline: the committed ridge-on-eval fit.
r0 = np.load(ROOT / "artifacts" / "ranker_0916.npz")
evaluate(lambda z: z @ r0["w"], test, "ridge on eval")

for hidden in (0, 32, 64):
    t0 = time.monotonic()
    fn, params = softmax_fit(hidden=hidden)
    label = "logit linear" if not hidden else f"logit MLP h={hidden}"
    evaluate(fn, test, f"{label} ({time.monotonic()-t0:.0f}s)")
    out = ROOT / "artifacts" / f"ranker_logit_h{hidden}.npz"
    if hidden:
        np.savez(out, mean=mean, sd=sd, w1=params[0], b1=params[1], w2=params[2])
    else:
        # The linear head is saved as a plain w so `eval_ranker.py --weights` reads it directly.
        np.savez(out, mean=mean, sd=sd, w=params[0][:, 0])
    print(f"    saved {out.name}")
