"""Trajectory collection from the bootstrap teacher.

Two kinds are collected: **for BC, imitating actions** (`collect_teacher_*`), and **for the value function, learning a board's value
from realized returns** (`collect_value_*`); the inputs and teacher signals differ.

The BC side is the `encode.py` observation vector -> the teacher's action. This is known to plateau at match 30%
(NOTES 'Investigated: BC does not reach 60-70% match').
The value side is the **post-drop board** of `features.py` -> **the points actually scored afterwards**.
The teacher's eval is not regressed because the inside of the tie band has been measured as truly indifferent
(NOTES 'Settled: the tie band really is indifferent'), so approximating eval would only hit the same ceiling
as the teacher. Only with realized returns as the label can an order enter the band.
"""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from os import PathLike

import numpy as np

from .agent import x_to_action
from .encode import encode
from .features import FEATURE_DIM, board_features
from .. import policy as pol
from ..policy import choose_x, rank_candidates
from ..reward import is_corner_watermelon
from ..sim.sim_env import SimEnv


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


# --- For the value function ---

# Thinning of moves whose candidate table is kept. Keeping every move adds 40+ candidates per move,
# 40x the raw trajectory. It is only used to verify rankings, so coarse is fine.
CANDIDATE_STRIDE = 8


@dataclass
class ValueDataset:
    """Training data for the value function. One row = 'the board after one move'.

    `returns` is the sum of real-game points scored **after** that board (undiscounted). The points of move t were
    earned on the way to board t, so they are not included. When ranking candidates it takes the form
    Q = (that move's real-game score) + V (post-drop board).

    Episodes with `truncated` set have missing returns. It does not just shift the mean:
    the values themselves come out small, so a flag is kept so the learner can drop them
    (the truncation item in NOTES 'How to measure').
    """

    feats: np.ndarray  # (T, FEATURE_DIM)
    rewards: np.ndarray  # (T,) real-game points earned on that move
    returns: np.ndarray  # (T,) sum of real-game points after that board
    steps: np.ndarray  # (T,) move number (0-based)
    episodes: np.ndarray  # (T,) episode number
    truncated: np.ndarray  # (T,) whether the move is from a truncated episode
    cornered: np.ndarray  # (T,) whether the move is from an episode that reached a corner watermelon
    # Candidate table (once every CANDIDATE_STRIDE moves). cand_row points to a row of feats.
    cand_feats: np.ndarray  # (M, FEATURE_DIM)
    cand_rewards: np.ndarray  # (M,) real-game points of that candidate
    cand_evals: np.ndarray  # (M,) the teacher's eval
    cand_chosen: np.ndarray  # (M,) whether the teacher chose the candidate
    cand_row: np.ndarray  # (M,)


def _empty_dataset() -> ValueDataset:
    feats = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    scalars = np.zeros(0, dtype=np.float32)
    ints = np.zeros(0, dtype=np.int32)
    flags = np.zeros(0, dtype=bool)
    return ValueDataset(
        feats, scalars, scalars, ints, ints, flags, flags,
        feats, scalars, scalars, flags, ints,
    )


def collect_value_episode(
    env: SimEnv,
    *,
    max_steps: int,
    episode: int = 0,
    candidate_stride: int = CANDIDATE_STRIDE,
) -> ValueDataset:
    """Value data for one game. Moves are chosen by the teacher (`choose_x`).

    The candidate table is taken from `rank_candidates`, and that same table is passed to `choose_x` to have it
    decide the move. This avoids running the physics twice; the move-selection rules are not copied here
    (a copy would drift from the original without anyone noticing).
    """
    obs = env.reset()
    feats: list[np.ndarray] = []
    rewards: list[float] = []
    steps: list[int] = []
    cand_feats: list[np.ndarray] = []
    cand_rewards: list[float] = []
    cand_evals: list[float] = []
    cand_chosen: list[bool] = []
    cand_row: list[int] = []
    cornered = False
    # Truncated only when leaving because the cap was reached. Cleared on paths that exit with break.
    truncated = True

    for step in range(max_steps):
        if obs.held_type is None:
            truncated = False
            break
        sign = pol._order_sign(list(obs.fruits))
        ranked = rank_candidates(obs)
        if not ranked:
            truncated = False
            break
        x = choose_x(obs, ranked=ranked)

        row = len(feats)
        if candidate_stride > 0 and step % candidate_stride == 0:
            for cand_eval, cand_x, after, cand_score in ranked:
                cand_feats.append(board_features(after, sign=sign))
                cand_rewards.append(cand_score)
                cand_evals.append(cand_eval)
                cand_chosen.append(cand_x == x)
                cand_row.append(row)

        result = env.step(x)
        feats.append(board_features(result.observation.fruits, sign=sign))
        rewards.append(result.score)
        steps.append(step)
        if is_corner_watermelon(result.observation.fruits):
            cornered = True
        obs = result.observation
        if result.done:
            truncated = False
            break

    n = len(feats)
    if n == 0:
        return _empty_dataset()

    # The value of board t is the sum of points scored after t. The points of move t are what it took to reach board t.
    returns = np.zeros(n, dtype=np.float32)
    running = 0.0
    for i in range(n - 1, -1, -1):
        returns[i] = running
        running += rewards[i]

    empty_feats = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return ValueDataset(
        feats=np.stack(feats).astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        returns=returns,
        steps=np.asarray(steps, dtype=np.int32),
        episodes=np.full(n, episode, dtype=np.int32),
        truncated=np.full(n, truncated, dtype=bool),
        cornered=np.full(n, cornered, dtype=bool),
        cand_feats=np.stack(cand_feats).astype(np.float32) if cand_row else empty_feats,
        cand_rewards=np.asarray(cand_rewards, dtype=np.float32),
        cand_evals=np.asarray(cand_evals, dtype=np.float32),
        cand_chosen=np.asarray(cand_chosen, dtype=bool),
        cand_row=np.asarray(cand_row, dtype=np.int32),
    )


def _concat(parts: list[ValueDataset]) -> ValueDataset:
    """Bundle per-episode results. cand_row is converted to row numbers after concatenation."""
    parts = [p for p in parts if len(p.feats)]
    if not parts:
        return _empty_dataset()
    rows = []
    offset = 0
    for part in parts:
        rows.append(part.cand_row + offset)
        offset += len(part.feats)
    return ValueDataset(
        feats=np.concatenate([p.feats for p in parts]),
        rewards=np.concatenate([p.rewards for p in parts]),
        returns=np.concatenate([p.returns for p in parts]),
        steps=np.concatenate([p.steps for p in parts]),
        episodes=np.concatenate([p.episodes for p in parts]),
        truncated=np.concatenate([p.truncated for p in parts]),
        cornered=np.concatenate([p.cornered for p in parts]),
        cand_feats=np.concatenate([p.cand_feats for p in parts]),
        cand_rewards=np.concatenate([p.cand_rewards for p in parts]),
        cand_evals=np.concatenate([p.cand_evals for p in parts]),
        cand_chosen=np.concatenate([p.cand_chosen for p in parts]),
        cand_row=np.concatenate(rows),
    )


def _collect_value_episode_job(
    seed: int, max_steps: int, episode: int, candidate_stride: int
) -> ValueDataset:
    """For ProcessPool. Placed at module top level (Windows spawn)."""
    env = SimEnv(seed=seed)
    return collect_value_episode(
        env, max_steps=max_steps, episode=episode, candidate_stride=candidate_stride
    )


def collect_value_episodes(
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    workers: int,
    log_every: int,
    candidate_stride: int = CANDIDATE_STRIDE,
) -> ValueDataset:
    """Collect value data in parallel and concatenate it. Seeds are assigned the same way as on the BC side."""
    jobs = [(seed + 1000 + ep, max_steps) for ep in range(1, episodes + 1)]

    if workers <= 1 or episodes <= 1:
        parts: list[ValueDataset] = []
        for i, (ep_seed, steps) in enumerate(jobs):
            parts.append(_collect_value_episode_job(ep_seed, steps, i, candidate_stride))
            if (i + 1) % log_every == 0 or i == 0 or i + 1 == episodes:
                rows = sum(len(p.feats) for p in parts)
                print(f"collect={i + 1:4d}/{episodes}  rows={rows}", flush=True)
        return _concat(parts)

    done = 0
    results: list[ValueDataset | None] = [None] * episodes
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_collect_value_episode_job, ep_seed, steps, i, candidate_stride): i
            for i, (ep_seed, steps) in enumerate(jobs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if done % log_every == 0 or done == 1 or done == episodes:
                rows = sum(len(r.feats) for r in results if r is not None)
                print(
                    f"collect={done:4d}/{episodes}  rows~{rows}  workers={workers}",
                    flush=True,
                )

    ordered: list[ValueDataset] = []
    for item in results:
        assert item is not None
        ordered.append(item)
    return _concat(ordered)


def save_value_dataset(data: ValueDataset, path: str | PathLike) -> None:
    """Save to npz. Read back with `np.load`."""
    np.savez_compressed(
        path,
        feats=data.feats,
        rewards=data.rewards,
        returns=data.returns,
        steps=data.steps,
        episodes=data.episodes,
        truncated=data.truncated,
        cornered=data.cornered,
        cand_feats=data.cand_feats,
        cand_rewards=data.cand_rewards,
        cand_evals=data.cand_evals,
        cand_chosen=data.cand_chosen,
        cand_row=data.cand_row,
    )
