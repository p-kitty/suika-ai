"""Trajectory collection from the bootstrap teacher."""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, as_completed

import numpy as np

from ..agent import x_to_action
from ..encode import encode
from ..policy import choose_x
from ..sim_env import SimEnv


def collect_teacher_episode(
    env: SimEnv,
    *,
    max_steps: int,
    pool: Executor | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Collect only teacher trajectories (no learning).

    Passing pool parallelizes choose_x candidate evaluation over processes. The side that spreads episodes
    themselves over a ProcessPool (_collect_teacher_episode_job) leaves this as None
    so processes are not started twice.
    """
    obs = env.reset()
    obs_list: list[np.ndarray] = []
    actions: list[int] = []

    for _ in range(max_steps):
        teacher_x = choose_x(obs, pool=pool)
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
    """For ProcessPool. Placed at module top level (Windows spawn).

    Processes are already used per episode, so the choose_x side
    pool is not passed here (avoids overusing cores through double parallelism).
    """
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
    """Collect teacher trajectories in parallel and concatenate them."""
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    jobs = [
        (seed + 1000 + ep, max_steps) for ep in range(1, episodes + 1)
    ]

    if workers <= 1 or episodes <= 1:
        # With no episode-level parallelism available, parallelize choose_x candidate evaluation
        # over processes instead (the pool is reused, since recreating it per move adds startup cost).
        with ProcessPoolExecutor() as move_pool:
            for i, (ep_seed, steps) in enumerate(jobs, start=1):
                env = SimEnv(seed=ep_seed)
                obs_list, actions = collect_teacher_episode(
                    env, max_steps=steps, pool=move_pool
                )
                obs_buf.extend(obs_list)
                act_buf.extend(actions)
                if i % log_every == 0 or i == 1 or i == episodes:
                    print(f"collect={i:4d}  buf={len(obs_buf)}", flush=True)
        return obs_buf, act_buf

    done = 0
    # Concatenate in episode order, not completion order, to match the buffer order of a serial run.
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
