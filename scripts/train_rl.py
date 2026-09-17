"""REINFORCE on the linear candidate ranker.

The ranker is already a conditional logit over the candidates of a position, so it is a stochastic policy
as it stands: pi(x) = softmax(w . z(x) / temp) over the post-drop features of each candidate. That makes
REINFORCE nineteen numbers wide, with no framework and no value network.

NOTES 'Planned: RL (REINFORCE)' gates this on a student that imitates the teacher and scores near it;
'What the student is missing is the second ply, not fidelity' is where that was reached (77% of bootstrap).
The reward is the real game's score, per the Training section -- dense penalties are not rewards.

Usage:
  python scripts/train_rl.py --eval-only --episodes 16 --temp 1.0
  python scripts/train_rl.py --check-gradient --episodes 128 --workers 8 --dump artifacts/rl_rollouts.npz
  python scripts/train_rl.py --sweep artifacts/rl_rollouts.npz
  python scripts/train_rl.py --iters 30 --batch 64 --workers 8 --lr 0.02
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


def _returns(rewards: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Discounted return-to-go per step."""
    out = np.empty(len(rewards), dtype=np.float64)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        out[i] = acc
    return out


def _episode_grads(rows: list[Rollout], baseline: str, gamma: float = GAMMA) -> np.ndarray:
    """(episodes x FEATURE_DIM) REINFORCE gradient contribution of each episode, not yet averaged.

    baseline "move" subtracts the batch's mean return-to-go at the same move number and divides by its SD.
    Return-to-go shrinks as a game goes on, so a single batch-wide mean makes the advantage mostly the move
    number (corr -0.766 measured); that cancels in expectation but not in a batch, and left the gradient as
    noise (NOTES 'Measured: REINFORCE on the ranker made it worse'). "batch" is the old one, kept only so
    --check-gradient can show the difference. "move-mean" subtracts the per-move mean without dividing by its
    SD: near the end only a few long games remain, their SD is small, and dividing inflates exactly those moves.
    """
    rets = [_returns(r.rewards, gamma) for r in rows]
    if baseline == "batch":
        flat = np.concatenate(rets)
        mu, sig = float(flat.mean()), float(flat.std()) + 1e-8
        advs = [(x - mu) / sig for x in rets]
    else:
        length = max(len(x) for x in rets)
        base = np.zeros(length)
        spread = np.ones(length)
        for t in range(length):
            vals = np.array([x[t] for x in rets if t < len(x)])
            base[t] = vals.mean()
            # With one game left at this move the SD is 0; do not blow its advantage up.
            if baseline == "move" and len(vals) > 1:
                spread[t] = vals.std() + 1e-8
        advs = [(x - base[: len(x)]) / spread[: len(x)] for x in rets]
    n_steps = max(1, sum(len(a) for a in advs))
    return np.array([r.grads.T @ a for r, a in zip(rows, advs)]) / n_steps


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _split_half(g_ep: np.ndarray, rng: np.random.Generator) -> float:
    """Cosine between the gradients of two random halves of the batch. Near 0 means the gradient is noise."""
    perm = rng.permutation(len(g_ep))
    half = len(g_ep) // 2
    return _cos(g_ep[perm[:half]].sum(0), g_ep[perm[half:]].sum(0))


def _signal_noise(g_ep: np.ndarray) -> tuple[float, float]:
    """(|E g|^2, tr Cov g) of one episode's gradient, estimated without bias from a batch.

    Resplitting one batch many ways and taking the median cosine does not measure how two INDEPENDENT batches
    agree: every split reuses the same episodes, so the median reflects how that one batch happened to fall. It
    read +0.377 at 32 a side, and training at 64 a side then drew -0.33, -0.22, +0.35, +0.01. This uses every
    episode once instead. |mean|^2 overstates |E g|^2 by tr(Cov)/n, so that is subtracted; the result can come
    out negative when there is no signal, which is the honest answer.
    """
    n = len(g_ep)
    mean = g_ep.mean(0)
    tr_cov = float(((g_ep - mean) ** 2).sum() / (n - 1))
    return float(mean @ mean) - tr_cov / n, tr_cov


def _predicted_agree(half: int, signal: float, noise: float) -> float:
    """Expected cosine between the summed gradients of two independent batches of `half` episodes each.

    Each sum is half*E[g] plus noise with trace half*tr(Cov), so the cosine is h s / (h s + tr) for s >= 0.
    """
    s = max(0.0, signal)
    return half * s / (half * s + noise) if noise > 0 else 1.0


def _report(label: str, g_ep: np.ndarray, rng: np.random.Generator) -> None:
    """One row: signal/noise per episode, predicted independent-batch cosine, and disjoint 8v8 pairs."""
    signal, noise = _signal_noise(g_ep)
    boot = [_signal_noise(g_ep[rng.integers(len(g_ep), size=len(g_ep))]) for _ in range(200)]
    cells = ""
    for h in (64, 256):
        dist = [_predicted_agree(h, bs, bn) for bs, bn in boot]
        cells += (f"   {_predicted_agree(h, signal, noise):+.2f} "
                  f"[{np.percentile(dist, 5):+.2f},{np.percentile(dist, 95):+.2f}]")
    perm = rng.permutation(len(g_ep))
    pairs = [(perm[k:k + 8], perm[k + 8:k + 16]) for k in range(0, len(g_ep) - 15, 16)]
    direct = float(np.mean([_cos(g_ep[a].sum(0), g_ep[b].sum(0)) for a, b in pairs]))
    print(f"  {label:<22} s/n {signal / noise:+.5f}{cells}   8v8 {direct:+.3f} ({len(pairs)} pairs)")


def _sweep(rows: list[Rollout], rng: np.random.Generator) -> None:
    print(f"  {len(rows)} episodes, {sum(len(r.rewards) for r in rows)} moves")
    print("  predicted cosine between independent batches of 64 / 256 episodes a side, 5-95% over resampled episodes")
    for gamma in (0.99, 0.995, 0.999, 1.0):
        for baseline in ("batch", "move", "move-mean"):
            _report(f"gamma {gamma:<5} {baseline}", _episode_grads(rows, baseline, gamma), rng)
    print("  a consistent direction is necessary, not sufficient: the batch baseline agreed and still made play worse")


def _save_rows(path: Path, rows: list[Rollout]) -> None:
    """Per-move rewards and (z_chosen - E z), so baselines and gamma can be swept without replaying."""
    lengths = [len(r.rewards) for r in rows]
    np.savez(path,
             seeds=np.array([r.seed for r in rows]), scores=np.array([r.score for r in rows]),
             offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
             rewards=np.concatenate([r.rewards for r in rows]).astype(np.float64),
             grads=np.concatenate([r.grads for r in rows]).astype(np.float32))


def _load_rows(path: Path) -> list[Rollout]:
    d = np.load(path)
    off = d["offsets"]
    return [Rollout(int(d["seeds"][i]), float(d["scores"][i]), int(off[i + 1] - off[i]),
                    d["grads"][off[i]:off[i + 1]].astype(np.float64), d["rewards"][off[i]:off[i + 1]])
            for i in range(len(off) - 1)]


def _play(seeds: list[int], w: np.ndarray, mean: np.ndarray, sd: np.ndarray,
          args: argparse.Namespace, workers: int, greedy: bool) -> list[Rollout]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [f.result() for f in as_completed(
            [pool.submit(_rollout, s, w, mean, sd, args.max_steps, args.temp, greedy) for s in seeds])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64, help="episodes per update")
    parser.add_argument("--episodes", type=int, default=16, help="--eval-only / --check-gradient episode count")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=940000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.02,
                        help="largest step as a fraction of |w|, taken only when the two halves fully agree")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--baseline", choices=("batch", "move", "move-mean"), default="move")
    parser.add_argument("--step-along", type=Path, nargs="+", default=None,
                        help="write one-step weights along the pooled gradient of these --dump files; plays nothing")
    parser.add_argument("--step-sizes", type=float, nargs="+", default=[0.01, 0.02, 0.04])
    parser.add_argument("--fixed-step", type=float, default=None,
                        help="take this fraction of |w| every update instead of gating on one split; "
                             "only after a pooled --sweep has shown the chosen gamma and baseline carry signal")
    parser.add_argument("--start", type=Path, default=START)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "ranker_rl.npz")
    parser.add_argument("--eval-only", action="store_true",
                        help="play the starting weights and report score, greedy and sampled")
    parser.add_argument("--dump", type=Path, default=None,
                        help="--check-gradient: save per-move rollouts so --sweep can reanalyse them without replaying")
    parser.add_argument("--sweep", type=Path, nargs="+", default=None,
                        help="reanalyse --dump files, pooled, across gamma and baselines; plays nothing. "
                             "128 episodes is too few: two dumps of that size swapped which baseline looked alive")
    parser.add_argument("--check-gradient", action="store_true",
                        help="play one batch from the starting weights and report split-half agreement "
                             "per baseline and half size, without updating anything")
    args = parser.parse_args()
    workers = resolve_workers(args.workers)

    d = np.load(args.start)
    w = (d["w"] if "w" in d.files else d["p0"][:, 0]).astype(np.float64)
    mean, sd = d["mean"].astype(np.float64), d["sd"].astype(np.float64)

    if args.eval_only:
        for greedy in (True, False):
            rows = _play([args.seed + i for i in range(args.episodes)], w, mean, sd, args, workers, greedy)
            sc = [r.score for r in rows]
            st = [float(r.steps) for r in rows]
            label = "greedy" if greedy else f"sampled temp={args.temp}"
            print(f"  {label:<22} score {statistics.mean(sc):8.1f}   steps {statistics.mean(st):6.1f}")
        return

    if args.step_along is not None:
        # One step from the starting weights along the gradient of rollouts already played with those weights.
        # On-policy for exactly those weights, so no new games are needed, and pooling 1024 episodes gives a
        # far better direction than a fresh batch of 256 would (predicted cosine with the true gradient ~0.67
        # against ~0.41 at gamma 1.0, per-move baseline). Each step size is saved for eval_ranker.py.
        rows = [row for path in args.step_along for row in _load_rows(path)]
        grad = _episode_grads(rows, args.baseline, args.gamma).sum(0)
        direction = grad / float(np.linalg.norm(grad))
        for size in args.step_sizes:
            out = args.out.with_name(f"{args.out.stem}_step{size:g}.npz")
            np.savez(out, w=w + size * float(np.linalg.norm(w)) * direction, mean=mean, sd=sd)
            print(f"  {len(rows)} episodes, gamma {args.gamma} {args.baseline}, step {size:g} of |w| -> {out}")
        return

    if args.sweep is not None:
        _sweep([row for path in args.sweep for row in _load_rows(path)], np.random.default_rng(0))
        return

    if args.check_gradient:
        rows = _play([args.seed + i for i in range(args.episodes)], w, mean, sd, args, workers, False)
        if args.dump is not None:
            _save_rows(args.dump, rows)
            print(f"  saved per-move rollouts to {args.dump}; rerun offline with --sweep {args.dump}")
        _sweep(rows, np.random.default_rng(0))
        return

    rng = np.random.default_rng(args.seed)
    seed = args.seed
    started = time.monotonic()
    for it in range(args.iters):
        rows = _play([seed + i for i in range(args.batch)], w, mean, sd, args, workers, False)
        seed += args.batch
        g_ep = _episode_grads(rows, args.baseline, args.gamma)
        grad = g_ep.sum(0)
        agree = _split_half(g_ep, rng)
        norm = float(np.linalg.norm(grad))
        if args.fixed_step is not None:
            # A single split of one batch scatters widely (four draws at 64 a side ran -0.33 to +0.35), so gating
            # every step on it mostly gates on luck. With --fixed-step the signal was established beforehand on a
            # pooled --sweep, and every update takes that size; agree is still printed to watch.
            step = args.fixed_step
        else:
            # A fixed-size normalised step walked the first run 16.3% along noise (30 random 2% steps give 11%),
            # so without a prior measurement a batch whose halves disagree moves nothing.
            step = args.lr * max(0.0, agree)
        if norm > 1e-12 and step > 0:
            w += step * float(np.linalg.norm(w)) * grad / norm
        sc = [r.score for r in rows]
        print(f"  iter {it + 1:>3}/{args.iters}  score {statistics.mean(sc):8.1f}"
              f"  median {statistics.median(sc):8.1f}  agree {agree:+.3f}  step {step:.4f}"
              f"  |w| {np.linalg.norm(w):.3f}  ({time.monotonic() - started:.0f}s)", flush=True)
        np.savez(args.out, w=w, mean=mean, sd=sd)
    print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
