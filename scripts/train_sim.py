"""sim 上で方策を学習する。

既定は bootstrap (`choose_x`) を先生にしたオフライン BC:
  1) 先生の軌道を集める (重みは触らない)
  2) 溜めたデータで何エポックか復習する
  3) student_r が最良の重みを保存

REINFORCE は match が十分上がってから手動で足す。

用法:
  python scripts/train_sim.py
  python scripts/train_sim.py --bc-episodes 80 --bc-epochs 60
  python scripts/train_sim.py --bc-episodes 80 --bc-epochs 60 --episodes 50 --lr 0.002
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import LinearPolicy, x_to_action
from src.encode import encode
from src.policy import choose_x
from src.sim_env import SimEnv

DEFAULT_CKPT = ROOT / "artifacts" / "policy_sim.npz"


def collect_teacher_episode(
    env: SimEnv,
    *,
    max_steps: int,
) -> tuple[list[np.ndarray], list[int]]:
    """先生軌道だけ集める (学習しない)。"""
    obs = env.reset()
    obs_list: list[np.ndarray] = []
    actions: list[int] = []

    for _ in range(max_steps):
        teacher_x = choose_x(obs)
        obs_list.append(encode(obs))
        actions.append(x_to_action(teacher_x))
        result = env.step(teacher_x)
        obs = result.observation
        if result.done:
            break
    return obs_list, actions


def match_rate(
    policy: LinearPolicy,
    obs_buf: list[np.ndarray],
    act_buf: list[int],
    *,
    rng: np.random.Generator,
    sample: int = 512,
) -> float:
    """バッファ上の greedy 一致率。"""
    n = len(obs_buf)
    if n == 0:
        return 0.0
    take = min(sample, n)
    idx = rng.choice(n, size=take, replace=False)
    hits = 0
    for i in idx:
        if int(policy.probs(obs_buf[i]).argmax()) == act_buf[i]:
            hits += 1
    return hits / take


def train_bc_epoch(
    policy: LinearPolicy,
    obs_buf: list[np.ndarray],
    act_buf: list[int],
    *,
    lr: float,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """1 エポック BC。平均 NLL。"""
    n = len(obs_buf)
    if n == 0:
        return 0.0
    order = np.arange(n)
    rng.shuffle(order)
    losses: list[float] = []
    for start in range(0, n, batch_size):
        batch = order[start : start + batch_size]
        loss = policy.bc_update(
            [obs_buf[i] for i in batch],
            [act_buf[i] for i in batch],
            lr=lr,
        )
        losses.append(loss)
    return statistics.fmean(losses)


def eval_student(
    policy: LinearPolicy,
    *,
    seed: int,
    episodes: int,
    max_steps: int,
) -> tuple[float, float]:
    """生徒 greedy の平均報酬・ステップ。"""
    rewards: list[float] = []
    steps_list: list[int] = []
    for i in range(episodes):
        env = SimEnv(seed=seed + i)
        obs = env.reset()
        total = 0.0
        steps = 0
        for _ in range(max_steps):
            _, x, _ = policy.act(obs, greedy=True)
            result = env.step(x)
            total += result.reward
            obs = result.observation
            steps += 1
            if result.done:
                break
        rewards.append(total)
        steps_list.append(steps)
    return statistics.fmean(rewards), statistics.fmean(steps_list)


def run_rl_episode(
    env: SimEnv,
    policy: LinearPolicy,
    *,
    max_steps: int,
    gamma: float,
    lr: float,
) -> tuple[float, int]:
    obs = env.reset()
    obs_list: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        action, x, vec = policy.act(obs)
        obs_list.append(vec)
        actions.append(action)
        result = env.step(x)
        rewards.append(result.reward)
        obs = result.observation
        if result.done:
            break

    returns: list[float] = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running

    baseline = statistics.fmean(returns) if returns else 0.0
    advantages = [r - baseline for r in returns]
    policy.update(obs_list, actions, advantages, lr=lr)
    return float(sum(rewards)), len(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-episodes", type=int, default=80)
    parser.add_argument("--bc-epochs", type=int, default=60)
    # RL は BC が足りてから。既定オフ。
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--bc-lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_CKPT,
        help="重み保存先 (nullptr で無効)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    policy = LinearPolicy(rng)
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []

    if args.bc_episodes > 0:
        print(
            f"=== collect teacher ({args.bc_episodes} ep) ===",
            flush=True,
        )
        for ep in range(1, args.bc_episodes + 1):
            env = SimEnv(seed=args.seed + 1000 + ep)
            obs_list, actions = collect_teacher_episode(
                env, max_steps=args.max_steps
            )
            obs_buf.extend(obs_list)
            act_buf.extend(actions)
            if ep % args.log_every == 0 or ep == 1 or ep == args.bc_episodes:
                print(
                    f"collect={ep:4d}  buf={len(obs_buf)}",
                    flush=True,
                )

    best_match = -1.0
    best_r = -float("inf")
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
                student_r, student_s = eval_student(
                    policy,
                    seed=args.seed + 9000,
                    episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                )
                # 実力 (student_r) 優先。同点なら match。
                better = student_r > best_r + 1e-9 or (
                    abs(student_r - best_r) <= 1e-9 and match > best_match
                )
                if better:
                    best_match = match
                    best_r = student_r
                    best_snap = policy.snapshot()
                print(
                    f"epoch={epoch:4d}  "
                    f"loss={loss:6.3f}  "
                    f"match={match:5.1%}  "
                    f"student_r={student_r:7.2f}  "
                    f"student_s={student_s:5.1f}  "
                    f"best_r={best_r:7.2f}  "
                    f"best_match={best_match:5.1%}",
                    flush=True,
                )

        if best_snap is not None:
            policy.restore(best_snap)
            print(
                f"restored best_r={best_r:.2f} best_match={best_match:.1%}",
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
            if ep % args.log_every == 0 or ep == 1:
                print(
                    f"rl={ep:4d}  "
                    f"reward={statistics.fmean(window):7.2f}  "
                    f"steps={statistics.fmean(window_steps):5.1f}",
                    flush=True,
                )
                window.clear()
                window_steps.clear()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        policy.save(args.save)
        print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
