"""Train a policy in the sim.

The default is offline BC with bootstrap (`choose_x`) as the teacher:
  1) collect the teacher's trajectories (weights untouched)
  2) review the stored data for several epochs
  3) save the weights with the best eval (= score - penalties)

Add REINFORCE by hand once match has risen enough.

Usage:
  python scripts/train_sim.py
  python scripts/train_sim.py --bc-episodes 100 --max-steps 100
  python scripts/train_sim.py --bc-episodes 100 --bc-epochs 80 --episodes 50 --lr 0.002
  python scripts/train_sim.py --bc-episodes 0 --bc-epochs 0 --episodes 400 --save artifacts/policy_rl.npz
  python scripts/train_sim.py --workers 8

Note: max-steps is not the game's losing condition but the truncation of collection and evaluation.
Until death or the cap. Longer learns more of the late game but collect gets slower.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT, ensure_import_path

ensure_import_path()

from src.agent import LinearPolicy
from src.parallel import default_workers
from src.sim_env import SimEnv
from src.training.bc import eval_student, match_rate, run_rl_episode, train_bc_epoch
from src.training.collect import collect_teacher_episodes

DEFAULT_CKPT = ROOT / "artifacts" / "policy_sim.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-episodes", type=int, default=100)
    parser.add_argument("--bc-epochs", type=int, default=80)
    # RL only after BC is enough. Off by default.
    parser.add_argument("--episodes", type=int, default=0)
    # Continue until game over. This is the cap for collection and evaluation.
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=None,
        help="move cap for student evaluation (--max-steps when omitted)",
    )
    parser.add_argument("--bc-lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel processes for teacher collection (logical cores/2 when omitted, 1 for serial)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_CKPT,
        help="where to save weights (nullptr to disable)",
    )
    args = parser.parse_args()
    eval_max_steps = (
        args.eval_max_steps if args.eval_max_steps is not None else args.max_steps
    )
    workers = (
        args.workers if args.workers is not None else default_workers()
    )

    rng = np.random.default_rng(args.seed)
    policy = LinearPolicy(rng)
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []

    if args.bc_episodes > 0:
        print(
            f"=== collect teacher ({args.bc_episodes} ep, "
            f"max_steps={args.max_steps}, workers={workers}) ===",
            flush=True,
        )
        obs_buf, act_buf = collect_teacher_episodes(
            episodes=args.bc_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            workers=workers,
            log_every=args.log_every,
        )

    best_match = -1.0
    best_eval = -float("inf")
    best_snap: dict[str, np.ndarray] | None = None

    if args.bc_epochs > 0 and obs_buf:
        print(
            f"=== BC train ({args.bc_epochs} epochs, n={len(obs_buf)}) ===",
            flush=True,
        )
        for epoch in range(1, args.bc_epochs + 1):
            loss = train_bc_epoch(
                policy,
                obs_buf,
                act_buf,
                lr=args.bc_lr,
                batch_size=args.batch_size,
                rng=rng,
            )
            if epoch % args.log_every == 0 or epoch == 1 or epoch == args.bc_epochs:
                match = match_rate(policy, obs_buf, act_buf, rng=rng)
                student_score, student_eval, student_s = eval_student(
                    policy,
                    seed=args.seed + 9000,
                    episodes=args.eval_episodes,
                    max_steps=eval_max_steps,
                )
                # Strength (eval) first. On ties, match.
                better = student_eval > best_eval + 1e-9 or (
                    abs(student_eval - best_eval) <= 1e-9 and match > best_match
                )
                if better:
                    best_match = match
                    best_eval = student_eval
                    best_snap = policy.snapshot()
                print(
                    f"epoch={epoch:4d}  "
                    f"loss={loss:6.3f}  "
                    f"match={match:5.1%}  "
                    f"score={student_score:7.2f}  "
                    f"eval={student_eval:8.2f}  "
                    f"student_s={student_s:5.1f}  "
                    f"best_eval={best_eval:8.2f}  "
                    f"best_match={best_match:5.1%}",
                    flush=True,
                )

        if best_snap is not None:
            policy.restore(best_snap)
            print(
                f"restored best_eval={best_eval:.2f} best_match={best_match:.1%}",
                flush=True,
            )

    if args.episodes > 0:
        print(f"=== REINFORCE ({args.episodes} ep, lr={args.lr}) ===", flush=True)
        window: list[float] = []
        window_steps: list[int] = []
        for ep in range(1, args.episodes + 1):
            env = SimEnv(seed=args.seed + 5000 + ep)
            total, steps = run_rl_episode(
                env,
                policy,
                max_steps=args.max_steps,
                gamma=args.gamma,
                lr=args.lr,
            )
            window.append(total)
            window_steps.append(steps)
            if ep % args.log_every == 0 or ep == 1 or ep == args.episodes:
                train_score = statistics.fmean(window)
                train_s = statistics.fmean(window_steps)
                student_score, student_eval, student_s = eval_student(
                    policy,
                    seed=args.seed + 9000,
                    episodes=args.eval_episodes,
                    max_steps=eval_max_steps,
                )
                if student_eval > best_eval + 1e-9:
                    best_eval = student_eval
                    best_snap = policy.snapshot()
                print(
                    f"rl={ep:4d}  "
                    f"train_score={train_score:7.2f}  "
                    f"train_s={train_s:5.1f}  "
                    f"score={student_score:7.2f}  "
                    f"eval={student_eval:8.2f}  "
                    f"student_s={student_s:5.1f}  "
                    f"best_eval={best_eval:8.2f}",
                    flush=True,
                )
                window.clear()
                window_steps.clear()

        if best_snap is not None:
            policy.restore(best_snap)
            print(f"restored best_eval={best_eval:.2f}", flush=True)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        policy.save(args.save)
        print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
