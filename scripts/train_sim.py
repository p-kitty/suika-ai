"""sim 上で方策を学習する。

既定は bootstrap (`choose_x`) を先生にしたオフライン BC:
  1) 先生の軌道を集める (重みは触らない)
  2) 溜めたデータで何エポックか復習する
  3) eval (= score - penalties) が最良の重みを保存

REINFORCE は match が十分上がってから手動で足す。

用法:
  python scripts/train_sim.py
  python scripts/train_sim.py --bc-episodes 100 --max-steps 100
  python scripts/train_sim.py --bc-episodes 100 --bc-epochs 80 --episodes 50 --lr 0.002
  python scripts/train_sim.py --bc-episodes 0 --bc-epochs 0 --episodes 400 --save artifacts/policy_rl.npz
  python scripts/train_sim.py --workers 8

注: max-steps はゲームの負け条件ではなく、収集・評価の打ち切り。
死ぬか上限に達するまで。長いほど終盤を学べるが collect は遅くなる。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def default_collect_workers() -> int:
    """CPU-bound の choose_x 向け。論理コアの半分を既定にする (9700X なら 8)。"""
    n = os.cpu_count() or 4
    return max(1, n // 2)


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


def _collect_teacher_episode_job(
    seed: int,
    max_steps: int,
) -> tuple[list[np.ndarray], list[int]]:
    """ProcessPool 用。モジュールトップレベルに置く (Windows spawn)。"""
    env = SimEnv(seed=seed)
    return collect_teacher_episode(env, max_steps=max_steps)


def collect_teacher_episodes(
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    workers: int,
    log_every: int,
) -> tuple[list[np.ndarray], list[int]]:
    """先生軌道を並列収集して結合する。"""
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    jobs = [
        (seed + 1000 + ep, max_steps) for ep in range(1, episodes + 1)
    ]

    if workers <= 1 or episodes <= 1:
        for i, (ep_seed, steps) in enumerate(jobs, start=1):
            obs_list, actions = _collect_teacher_episode_job(ep_seed, steps)
            obs_buf.extend(obs_list)
            act_buf.extend(actions)
            if i % log_every == 0 or i == 1 or i == episodes:
                print(f"collect={i:4d}  buf={len(obs_buf)}", flush=True)
        return obs_buf, act_buf

    done = 0
    # 完了順ではなく ep 番号順で結合して、直列時と同じバッファ順にする。
    results: list[tuple[list[np.ndarray], list[int]] | None] = [None] * episodes
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_collect_teacher_episode_job, ep_seed, steps): i
            for i, (ep_seed, steps) in enumerate(jobs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if done % log_every == 0 or done == 1 or done == episodes:
                filled = sum(len(r[0]) for r in results if r is not None)
                print(
                    f"collect={done:4d}/{episodes}  buf~{filled}  workers={workers}",
                    flush=True,
                )

    for item in results:
        assert item is not None
        obs_list, actions = item
        obs_buf.extend(obs_list)
        act_buf.extend(actions)
    return obs_buf, act_buf


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
) -> tuple[float, float, float]:
    """生徒 greedy の平均 score・平均 eval・平均ステップ。"""
    scores: list[float] = []
    evals: list[float] = []
    steps_list: list[int] = []
    for i in range(episodes):
        env = SimEnv(seed=seed + i)
        obs = env.reset()
        total_score = 0.0
        total_eval = 0.0
        steps = 0
        for _ in range(max_steps):
            _, x, _ = policy.act(obs, greedy=True)
            result = env.step(x)
            total_score += result.score
            total_eval += result.eval_score
            obs = result.observation
            steps += 1
            if result.done:
                break
        scores.append(total_score)
        evals.append(total_eval)
        steps_list.append(steps)
    return (
        statistics.fmean(scores),
        statistics.fmean(evals),
        statistics.fmean(steps_list),
    )


def run_rl_episode(
    env: SimEnv,
    policy: LinearPolicy,
    *,
    max_steps: int,
    gamma: float,
    lr: float,
) -> tuple[float, int]:
    """報酬は本家点 (score) のみ。密な減点は報酬にしない。"""
    obs = env.reset()
    obs_list: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        action, x, vec = policy.act(obs)
        obs_list.append(vec)
        actions.append(action)
        result = env.step(x)
        rewards.append(result.score)
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
    parser.add_argument("--bc-episodes", type=int, default=100)
    parser.add_argument("--bc-epochs", type=int, default=80)
    # RL は BC が足りてから。既定オフ。
    parser.add_argument("--episodes", type=int, default=0)
    # ゲームオーバーまでは続行。これは収集・評価の上限打ち切り。
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=None,
        help="生徒評価の手数上限 (省略時は --max-steps)",
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
        help="教師収集の並列プロセス数 (省略時は論理コア/2、1 で直列)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_CKPT,
        help="重み保存先 (nullptr で無効)",
    )
    args = parser.parse_args()
    eval_max_steps = (
        args.eval_max_steps if args.eval_max_steps is not None else args.max_steps
    )
    workers = (
        args.workers if args.workers is not None else default_collect_workers()
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
                # 実力 (eval) 優先。同点なら match。
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
