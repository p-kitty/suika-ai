"""sim 上で方策を学習する。

既定は bootstrap (`choose_x`) を先生にした BC のみ。
先生データを溜めて何度も復習し、生徒自身の greedy 報酬で見る。
REINFORCE は match が十分上がってから手動で足す。

用法:
  python scripts/train_sim.py
  python scripts/train_sim.py --bc-episodes 200 --replay 8
  python scripts/train_sim.py --bc-episodes 200 --episodes 50 --lr 0.002
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
    policy: LinearPolicy,
    *,
    max_steps: int,
) -> tuple[list[np.ndarray], list[int], float, int]:
    """先生軌道を集める。match は更新前の greedy 一致率。"""
    obs = env.reset()
    obs_list: list[np.ndarray] = []
    actions: list[int] = []
    matches = 0
    steps = 0

    for _ in range(max_steps):
        teacher_x = choose_x(obs)
        teacher_a = x_to_action(teacher_x)
        vec = encode(obs)
        if int(policy.probs(vec).argmax()) == teacher_a:
            matches += 1
        obs_list.append(vec)
        actions.append(teacher_a)
        result = env.step(teacher_x)
        obs = result.observation
        steps += 1
        if result.done:
            break

    match = matches / steps if steps else 0.0
    return obs_list, actions, match, steps


def bc_replay(
    policy: LinearPolicy,
    obs_buf: list[np.ndarray],
    act_buf: list[int],
    *,
    fresh_obs: list[np.ndarray],
    fresh_act: list[int],
    lr: float,
    replay: int,
    batch_size: int,
    rng: np.random.Generator,
) -> None:
    """今エピソードを学習し、バッファからミニバッチを数回復習。"""
    if fresh_obs:
        policy.bc_update(fresh_obs, fresh_act, lr=lr)
    n = len(obs_buf)
    if n == 0 or replay <= 0:
        return
    take = min(batch_size, n)
    for _ in range(replay):
        batch = rng.choice(n, size=take, replace=False)
        policy.bc_update(
            [obs_buf[i] for i in batch],
            [act_buf[i] for i in batch],
            lr=lr,
        )


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
    parser.add_argument("--bc-episodes", type=int, default=200)
    # RL は BC が足りてから。既定オフ。
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--bc-lr", type=float, default=0.02)
    parser.add_argument("--replay", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_CKPT,
        help="重み保存先 (npz で無効)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    policy = LinearPolicy(rng)
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []

    if args.bc_episodes > 0:
        print(
            f"=== BC from bootstrap ({args.bc_episodes} ep, "
            f"replay={args.replay}) ===",
            flush=True,
        )
        window_m: list[float] = []
        for ep in range(1, args.bc_episodes + 1):
            env = SimEnv(seed=args.seed + 1000 + ep)
            obs_list, actions, match, _ = collect_teacher_episode(
                env, policy, max_steps=args.max_steps
            )
            obs_buf.extend(obs_list)
            act_buf.extend(actions)
            bc_replay(
                policy,
                obs_buf,
                act_buf,
                fresh_obs=obs_list,
                fresh_act=actions,
                lr=args.bc_lr,
                replay=args.replay,
                batch_size=args.batch_size,
                rng=rng,
            )
            window_m.append(match)
            if ep % args.log_every == 0 or ep == 1:
                student_r, student_s = eval_student(
                    policy,
                    seed=args.seed + 9000,
                    episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                )
                print(
                    f"bc={ep:4d}  "
                    f"match={statistics.fmean(window_m):5.1%}  "
                    f"student_r={student_r:7.2f}  "
                    f"student_s={student_s:5.1f}  "
                    f"buf={len(obs_buf)}",
                    flush=True,
                )
                window_m.clear()

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
