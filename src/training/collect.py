"""bootstrap 先生の軌道収集。

2 種類を集める。**行動を真似る BC 用**（`collect_teacher_*`）と、**盤の値を
実リターンから覚える価値関数用**（`collect_value_*`）で、入力も教師信号も別。

BC 側は `encode.py` の観測ベクトル -> 教師の行動。これは match 30% で頭打ちに
なることが分かっている（NOTES「BC が match 60-70% に届かない」）。
価値側は `features.py` の**落下後の盤** -> **その先で実際に取った点**。
教師の eval を回帰しないのは、同点帯の中が本当に無差別だと測れているので
（NOTES「決着: 同点帯は本当に無差別」）、eval を近似しても教師と同じ天井に
ぶつかるだけだから。ラベルを実リターンにして初めて帯の中に順序が入りうる。
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
    """先生軌道だけ集める (学習しない)。

    pool を渡すと choose_x の候補評価をプロセス並列化する。エピソード自体を
    ProcessPool へ分散する側 (_collect_teacher_episode_job) では、二重に
    プロセスを起動しないようこちらは None のままにする。
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
    """ProcessPool 用。モジュールトップレベルに置く (Windows spawn)。

    エピソード単位で既にプロセスを使っているので、ここでは choose_x 側の
    pool は渡さない (二重並列化によるコア過剰使用を避ける)。
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
    """先生軌道を並列収集して結合する。"""
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    jobs = [
        (seed + 1000 + ep, max_steps) for ep in range(1, episodes + 1)
    ]

    if workers <= 1 or episodes <= 1:
        # エピソード単位の並列化先が無いぶん、代わりに choose_x の候補評価を
        # プロセス並列化する (手ごとにプールを作り直すと起動コストが乗るので使い回す)。
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


# --- 価値関数用 ---

# 候補表を残す手の間引き。全手ぶん残すと 1 手あたり候補 40 本超が乗り、素の
# 軌道の 40 倍になる。順位の検証にしか使わないので粗くてよい。
CANDIDATE_STRIDE = 8


@dataclass
class ValueDataset:
    """価値関数の学習データ。1 行 =「1 手打ったあとの盤」。

    `returns` はその盤から**先**で取った本家点の総和（割引なし）。手 t の点は
    盤 t に着くまでに得たものなので入れない。候補を並べるときは
    Q =（その手の本家点）+ V（落下後の盤）の形になる。

    `truncated` が立ったエピソードはリターンが欠けている。平均が寄るだけでは
    済まず値そのものが小さく出るので、学習側で落とせるようフラグで持つ
    （NOTES「測定のしかた」の打ち切りの項）。
    """

    feats: np.ndarray  # (T, FEATURE_DIM)
    rewards: np.ndarray  # (T,) その手で得た本家点
    merges: np.ndarray  # (T,) その手の合体数。cascades (3 以上) の材料
    returns: np.ndarray  # (T,) その盤から先の本家点の総和
    steps: np.ndarray  # (T,) 手番 (0 始まり)
    episodes: np.ndarray  # (T,) エピソード番号
    truncated: np.ndarray  # (T,) 打ち切りエピソードの手か
    cornered: np.ndarray  # (T,) 角スイカに到達したエピソードの手か
    # 候補表 (CANDIDATE_STRIDE 手に 1 回)。cand_row が feats の行を指す。
    cand_feats: np.ndarray  # (M, FEATURE_DIM)
    cand_rewards: np.ndarray  # (M,) その候補の本家点
    cand_evals: np.ndarray  # (M,) 教師の eval
    cand_chosen: np.ndarray  # (M,) 教師が選んだ候補か
    cand_row: np.ndarray  # (M,)


def _empty_dataset() -> ValueDataset:
    feats = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    scalars = np.zeros(0, dtype=np.float32)
    ints = np.zeros(0, dtype=np.int32)
    flags = np.zeros(0, dtype=bool)
    return ValueDataset(
        feats, scalars, ints, scalars, ints, ints, flags, flags,
        feats, scalars, scalars, flags, ints,
    )


def collect_value_episode(
    env: SimEnv,
    *,
    max_steps: int,
    episode: int = 0,
    candidate_stride: int = CANDIDATE_STRIDE,
) -> ValueDataset:
    """1 局ぶんの価値データ。手は教師 (`choose_x`) が選ぶ。

    候補表は `rank_candidates` から取り、その同じ表を `choose_x` に渡して手を
    決めさせる。物理を二度回さないためで、手選びの規則をここへ写さない
    （写すと本家とずれても気づけない）。
    """
    obs = env.reset()
    feats: list[np.ndarray] = []
    rewards: list[float] = []
    merges: list[int] = []
    steps: list[int] = []
    cand_feats: list[np.ndarray] = []
    cand_rewards: list[float] = []
    cand_evals: list[float] = []
    cand_chosen: list[bool] = []
    cand_row: list[int] = []
    cornered = False
    # 上限に達して抜けたときだけ打ち切り。break で出た経路では下ろす。
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
        merges.append(result.merges)
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

    # 盤 t の値は t より後で取った点の総和。手 t の点は盤 t に着くまでの分。
    returns = np.zeros(n, dtype=np.float32)
    running = 0.0
    for i in range(n - 1, -1, -1):
        returns[i] = running
        running += rewards[i]

    empty_feats = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return ValueDataset(
        feats=np.stack(feats).astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        merges=np.asarray(merges, dtype=np.int32),
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
    """エピソードごとの結果を束ねる。cand_row は結合後の行番号へ直す。"""
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
        merges=np.concatenate([p.merges for p in parts]),
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
    """ProcessPool 用。モジュールトップレベルに置く (Windows spawn)。"""
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
    """価値データを並列収集して結合する。seed の振り方は BC 側と揃えてある。"""
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
    """npz へ保存。読み出しは `np.load`。"""
    np.savez_compressed(
        path,
        feats=data.feats,
        rewards=data.rewards,
        merges=data.merges,
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
