"""sim 上で方策を学習する。

用法:
  python scripts/train_sim.py bc
  python scripts/train_sim.py bc --max-steps 300 --episodes 100
  python scripts/train_sim.py rl
  python scripts/train_sim.py rl --load artifacts/policy_sim.npz

max-steps は打ち切り上限 (負けラインではない)。
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT, ensure_import_path

ensure_import_path()

from src.agent import LinearPolicy
from src.parallel import default_workers
from src.training.bc import eval_student, match_rate, run_rl_batch, train_bc_epoch
from src.training.collect import collect_teacher_episodes

DEFAULT_BC_CKPT = ROOT / "artifacts" / "policy_sim.npz"
DEFAULT_RL_CKPT = ROOT / "artifacts" / "policy_sim_rl.npz"


@dataclass(frozen=True)
class SharedConfig:
    max_steps: int
    seed: int
    log_every: int
    eval_episodes: int
    save: Path


@dataclass(frozen=True)
class BcConfig(SharedConfig):
    collect_episodes: int
    epochs: int
    workers: int
    bc_lr: float
    batch_size: int


@dataclass(frozen=True)
class RlConfig(SharedConfig):
    load: Path
    episodes: int
    lr: float
    rl_batch: int
    entropy_coef: float
    entropy_decay: float
    lr_decay: float
    gamma: float


def _is_better(
    score: float,
    steps: float,
    match: float,
    *,
    best_score: float,
    best_steps: float,
    best_match: float,
) -> bool:
    if score > best_score + 1e-9:
        return True
    if abs(score - best_score) > 1e-9:
        return False
    if steps > best_steps + 1e-9:
        return True
    if abs(steps - best_steps) > 1e-9:
        return False
    return match > best_match


def run_bc(policy: LinearPolicy, cfg: BcConfig, rng: np.random.Generator) -> None:
    workers = cfg.workers if cfg.workers > 0 else default_workers()
    print(
        f"=== collect teacher ({cfg.collect_episodes} ep, "
        f"max_steps={cfg.max_steps}, workers={workers}) ===",
        flush=True,
    )
    obs_buf, act_buf = collect_teacher_episodes(
        episodes=cfg.collect_episodes,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
        workers=workers,
        log_every=cfg.log_every,
    )

    best_score = -float("inf")
    best_steps = -1.0
    best_match = -1.0
    best_snap: dict[str, np.ndarray] | None = None

    print(f"=== BC train ({cfg.epochs} epochs, n={len(obs_buf)}) ===", flush=True)
    for epoch in range(1, cfg.epochs + 1):
        loss = train_bc_epoch(
            policy,
            obs_buf,
            act_buf,
            lr=cfg.bc_lr,
            batch_size=cfg.batch_size,
            rng=rng,
        )
        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            match = match_rate(policy, obs_buf, act_buf, rng=rng)
            student_score, student_s = eval_student(
                policy,
                seed=cfg.seed + 9000,
                episodes=cfg.eval_episodes,
                max_steps=cfg.max_steps,
            )
            if _is_better(
                student_score,
                student_s,
                match,
                best_score=best_score,
                best_steps=best_steps,
                best_match=best_match,
            ):
                best_score = student_score
                best_steps = student_s
                best_match = match
                best_snap = policy.snapshot()
            print(
                f"epoch={epoch:4d}  "
                f"loss={loss:6.3f}  "
                f"match={match:5.1%}  "
                f"score={student_score:7.2f}  "
                f"student_s={student_s:5.1f}  "
                f"best_score={best_score:7.2f}  "
                f"best_match={best_match:5.1%}",
                flush=True,
            )

    if best_snap is not None:
        policy.restore(best_snap)
        print(
            f"restored best_score={best_score:.2f} "
            f"best_s={best_steps:.1f} best_match={best_match:.1%}",
            flush=True,
        )


def run_rl(policy: LinearPolicy, cfg: RlConfig) -> None:
    best_score = -float("inf")
    best_steps = -1.0
    best_snap: dict[str, np.ndarray] | None = None

    base_score, base_s = eval_student(
        policy,
        seed=cfg.seed + 9000,
        episodes=cfg.eval_episodes,
        max_steps=cfg.max_steps,
    )
    best_score = base_score
    best_steps = base_s
    best_snap = policy.snapshot()
    print(
        f"RL baseline  score={base_score:.2f}  student_s={base_s:.1f}",
        flush=True,
    )

    print(
        f"=== REINFORCE ({cfg.episodes} ep, batch={cfg.rl_batch}, "
        f"lr={cfg.lr}, entropy={cfg.entropy_coef}) ===",
        flush=True,
    )
    window: list[float] = []
    window_steps: list[int] = []
    rl_lr = cfg.lr
    entropy = cfg.entropy_coef
    batch_num = 0
    ep = 0
    while ep < cfg.episodes:
        batch_num += 1
        n = min(cfg.rl_batch, cfg.episodes - ep)
        seeds = [cfg.seed + 5000 + ep + i for i in range(n)]
        ep += n
        train_score, train_s, loss = run_rl_batch(
            policy,
            seeds=seeds,
            max_steps=cfg.max_steps,
            gamma=cfg.gamma,
            lr=rl_lr,
            entropy_coef=entropy,
        )
        rl_lr *= cfg.lr_decay
        entropy *= cfg.entropy_decay
        window.append(train_score)
        window_steps.append(train_s)
        if batch_num == 1 or ep % cfg.log_every == 0 or ep == cfg.episodes:
            student_score, student_s = eval_student(
                policy,
                seed=cfg.seed + 9000,
                episodes=cfg.eval_episodes,
                max_steps=cfg.max_steps,
            )
            if _is_better(
                student_score,
                student_s,
                0.0,
                best_score=best_score,
                best_steps=best_steps,
                best_match=0.0,
            ):
                best_score = student_score
                best_steps = student_s
                best_snap = policy.snapshot()
            print(
                f"rl={ep:4d}  "
                f"batch={batch_num:3d}  "
                f"loss={loss:7.3f}  "
                f"train_score={statistics.fmean(window):7.2f}  "
                f"train_s={statistics.fmean(window_steps):5.1f}  "
                f"score={student_score:7.2f}  "
                f"student_s={student_s:5.1f}  "
                f"best_score={best_score:7.2f}  "
                f"lr={rl_lr:.5f}  ent={entropy:.4f}",
                flush=True,
            )
            window.clear()
            window_steps.clear()

    if best_snap is not None:
        policy.restore(best_snap)
        print(
            f"restored best_score={best_score:.2f} best_s={best_steps:.1f}",
            flush=True,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    bc = sub.add_parser("bc", help="教師収集 + BC")
    bc.add_argument("--max-steps", type=int, default=300, help="手数上限 (既定 300)")
    bc.add_argument("--seed", type=int, default=0)
    bc.add_argument(
        "--episodes",
        type=int,
        default=100,
        metavar="N",
        help="教師収集 ep (既定 100)",
    )
    bc.add_argument(
        "--epochs",
        type=int,
        default=80,
        metavar="N",
        help="BC epoch (既定 80)",
    )
    bc.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_BC_CKPT,
        help=f"保存先 (既定 {DEFAULT_BC_CKPT.name})",
    )
    bc.add_argument("--workers", type=int, default=0, help="並列数 (0=自動)")
    tune_bc = bc.add_argument_group("詳細 (通常は触らない)")
    tune_bc.add_argument("--log-every", type=int, default=10)
    tune_bc.add_argument("--eval-episodes", type=int, default=8)
    tune_bc.add_argument("--bc-lr", type=float, default=0.05)
    tune_bc.add_argument("--batch-size", type=int, default=64)

    rl = sub.add_parser("rl", help="BC 済み npz から REINFORCE")
    rl.add_argument("--max-steps", type=int, default=300, help="手数上限 (既定 300)")
    rl.add_argument("--seed", type=int, default=0)
    rl.add_argument(
        "--load",
        type=Path,
        default=DEFAULT_BC_CKPT,
        help=f"入力 npz (既定 {DEFAULT_BC_CKPT.name})",
    )
    rl.add_argument(
        "--episodes",
        type=int,
        default=400,
        metavar="N",
        help="RL ep (既定 400)",
    )
    rl.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_RL_CKPT,
        help=f"保存先 (既定 {DEFAULT_RL_CKPT.name})",
    )
    tune_rl = rl.add_argument_group("詳細 (通常は触らない)")
    tune_rl.add_argument("--log-every", type=int, default=10)
    tune_rl.add_argument("--eval-episodes", type=int, default=16)
    tune_rl.add_argument("--lr", type=float, default=0.001)
    tune_rl.add_argument("--rl-batch", type=int, default=16)
    tune_rl.add_argument("--entropy-coef", type=float, default=0.02)
    tune_rl.add_argument("--entropy-decay", type=float, default=0.995)
    tune_rl.add_argument("--lr-decay", type=float, default=0.995)
    tune_rl.add_argument("--gamma", type=float, default=0.99)

    return parser


def _legacy_argv(argv: list[str]) -> list[str]:
    """旧 --bc-episodes / --load + --episodes をサブコマンドへ。"""
    if not argv or argv[0] in ("bc", "rl"):
        return argv
    if not argv[0].startswith("-"):
        return argv

    if any(a.startswith("--bc-") for a in argv):
        out: list[str] = ["bc"]
        i = 0
        while i < len(argv):
            if argv[i] == "--bc-episodes" and i + 1 < len(argv):
                out.extend(["--episodes", argv[i + 1]])
                i += 2
            elif argv[i] == "--bc-epochs" and i + 1 < len(argv):
                out.extend(["--epochs", argv[i + 1]])
                i += 2
            else:
                out.append(argv[i])
                i += 1
        return out

    if "--load" in argv:
        return ["rl", *argv]

    return ["bc", *argv]


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(_legacy_argv(raw))

    rng = np.random.default_rng(args.seed)
    policy = LinearPolicy(rng)

    if args.mode == "rl":
        if not args.load.is_file():
            raise SystemExit(f"checkpoint が無い: {args.load}")
        policy.load(args.load)
        print(f"loaded {args.load}", flush=True)
        run_rl(
            policy,
            RlConfig(
                max_steps=args.max_steps,
                seed=args.seed,
                log_every=args.log_every,
                eval_episodes=args.eval_episodes,
                save=args.save,
                load=args.load,
                episodes=args.episodes,
                lr=args.lr,
                rl_batch=args.rl_batch,
                entropy_coef=args.entropy_coef,
                entropy_decay=args.entropy_decay,
                lr_decay=args.lr_decay,
                gamma=args.gamma,
            ),
        )
    else:
        run_bc(
            policy,
            BcConfig(
                max_steps=args.max_steps,
                seed=args.seed,
                log_every=args.log_every,
                eval_episodes=args.eval_episodes,
                save=args.save,
                collect_episodes=args.episodes,
                epochs=args.epochs,
                workers=args.workers,
                bc_lr=args.bc_lr,
                batch_size=args.batch_size,
            ),
            rng,
        )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.save)
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
