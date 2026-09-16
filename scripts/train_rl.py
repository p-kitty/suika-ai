"""REINFORCE on the linear candidate ranker.

The ranker is already a conditional logit over the candidates of a position, so it is a stochastic policy
as it stands: pi(x) = softmax(w . z(x) / temp) over the post-drop features of each candidate. That makes
REINFORCE nineteen numbers wide, with no framework and no value network.

NOTES 'Planned: RL (REINFORCE)' gates this on a student that imitates the teacher and scores near it;
'What the student is missing is the second ply, not fidelity' is where that was reached (77% of bootstrap).
The reward is the real game's score, per the Training section -- dense penalties are not rewards.

Usage:
  python scripts/train_rl.py --eval-only --episodes 16 --temp 1.0
  python scripts/train_rl.py --iters 30 --batch 32 --workers 8 --lr 0.02
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import NamedTuple
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT
from src import policy as pol
from src.observe import Observation
from src.sim.sim_env import SimEnv
from src.training.features import FEATURE_DIM, board_features
from src.util.parallel import resolve_workers

START = ROOT / "artifacts" / "ranker_logit_h0.npz"
# Return-to-go is discounted so a move is credited with what follows it rather than with the whole game.
# 1.0 would make every move of an episode share one number and drown the signal in draw luck.
GAMMA = 0.99


def _features(obs: Observation) -> tuple[np.ndarray, list[float]]:
    """(candidates x FEATURE_DIM) post-drop features and their drop columns."""
    ranked = pol.rank_candidates(obs)
    if not ranked:
        return np.zeros((0, FEATURE_DIM), dtype=np.float64), []
    sign = pol._order_sign(list(obs.fruits))
    feats = np.empty((len(ranked), FEATURE_DIM), dtype=np.float64)
    xs: list[float] = []
    for i, (_e, x, after, _s) in enumerate(ranked):
        feats[i] = board_features(after, sign=sign)
        xs.append(x)
    return feats, xs


class Rollout(NamedTuple):
    """One episode: its score, and the per-step (z_chosen - E[z]) and rewards REINFORCE needs."""

    seed: int
    score: float
    steps: int
    grads: np.ndarray
    rewards: np.ndarray


def _rollout(seed: int, w: np.ndarray, mean: np.ndarray, sd: np.ndarray,
             max_steps: int, temp: float, greedy: bool) -> Rollout:
    """Play one episode. Returns the score and the per-step (z_chosen - E[z]) needed by REINFORCE."""
    rng = np.random.default_rng(seed ^ 0x5EED)
    env = SimEnv(seed=seed)
    obs = env.reset()
    grads: list[np.ndarray] = []
    rewards: list[float] = []
    score = 0.0
    steps = 0
    for _ in range(max_steps):
        if obs.held_type is None:
            break
        feats, xs = _features(obs)
        if not xs:
            break
        z = (feats - mean) / sd
        logits = (z @ w) / temp
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        k = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        # d/dw log pi(k) = z_k - sum_j p_j z_j
        grads.append(z[k] - p @ z)
        result = env.step(xs[k])
        rewards.append(float(result.score))
        score += result.score
        steps += 1
        obs = result.observation
        if result.done:
            break
    return Rollout(seed, score, int(steps), np.asarray(grads), np.asarray(rewards))


def _returns(rewards: np.ndarray) -> np.ndarray:
    """Discounted return-to-go per step."""
    out = np.empty_like(rewards)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + GAMMA * acc
        out[i] = acc
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch", type=int, default=48, help="episodes per update")
    parser.add_argument("--episodes", type=int, default=16, help="--eval-only episode count")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=940000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.02,
                        help="step as a fraction of |w| per update, not a raw learning rate")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--start", type=Path, default=START)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "ranker_rl.npz")
    parser.add_argument("--eval-only", action="store_true",
                        help="play the starting weights and report score, greedy and sampled")
    args = parser.parse_args()
    workers = resolve_workers(args.workers)

    d = np.load(args.start)
    w = (d["w"] if "w" in d.files else d["p0"][:, 0]).astype(np.float64)
    mean, sd = d["mean"].astype(np.float64), d["sd"].astype(np.float64)

    if args.eval_only:
        for greedy in (True, False):
            seeds = [args.seed + i for i in range(args.episodes)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                rows = [f.result() for f in as_completed(
                    [pool.submit(_rollout, s, w, mean, sd, args.max_steps, args.temp, greedy)
                     for s in seeds])]
            sc = [r.score for r in rows]
            st = [float(r.steps) for r in rows]
            label = "greedy" if greedy else f"sampled temp={args.temp}"
            print(f"  {label:<22} score {statistics.mean(sc):8.1f}   steps {statistics.mean(st):6.1f}")
        return

    seed = args.seed
    started = time.monotonic()
    for it in range(args.iters):
        seeds = [seed + i for i in range(args.batch)]
        seed += args.batch
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = [f.result() for f in as_completed(
                [pool.submit(_rollout, s, w, mean, sd, args.max_steps, args.temp, False)
                 for s in seeds])]
        # Advantage: return-to-go standardised over the whole batch. The baseline is what keeps the
        # update from chasing draw luck (per-game SD is ~600, NOTES 'How to measure').
        all_ret = np.concatenate([_returns(r.rewards) for r in rows])
        mu, sig = all_ret.mean(), all_ret.std() + 1e-8
        grad = np.zeros(FEATURE_DIM)
        n_steps = 0
        for r in rows:
            g = r.grads
            adv = (_returns(r.rewards) - mu) / sig
            grad += g.T @ adv
            n_steps += len(adv)
        grad /= max(1, n_steps)
        # Normalised step: move |w| by a fixed fraction along the gradient. A raw lr has to be
        # guessed against |grad|, and the first run guessed four orders of magnitude low --
        # |w|=10.6 moved 0.0004 over five iterations, so the flat scores were an unchanged policy.
        norm = float(np.linalg.norm(grad))
        if norm > 1e-12:
            w += args.lr * float(np.linalg.norm(w)) * grad / norm
        sc = [r.score for r in rows]
        print(f"  iter {it + 1:>3}/{args.iters}  score {statistics.mean(sc):8.1f}"
              f"  median {statistics.median(sc):8.1f}  |w| {np.linalg.norm(w):.3f}"
              f"  ({time.monotonic() - started:.0f}s)", flush=True)
        np.savez(args.out, w=w, mean=mean, sd=sd)
    print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
