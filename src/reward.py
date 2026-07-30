"""学習用の報酬。ダブルスイカを消したら成功終了が目的。"""

from __future__ import annotations

from .observe import Observation
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT

WATERMELON = MAX_FRUIT_TYPE
# この y より上に頭頂が出たら負け (y は下向き)。
GAME_OVER_Y = 40.0

# 生存 1 手。
STEP_REWARD = 0.05
# 合成 1 回あたり (大きい実ほど重い)。
MERGE_WEIGHT = 1.0
# 盤上の最大 type が伸びたとき。
PROGRESS_WEIGHT = 2.0
# スイカが新たに増えたとき。
WATERMELON_BONUS = 20.0
# スイカが 2 個以上になった瞬間 (キープ加点ではない)。
DOUBLE_REACH_BONUS = 15.0
# スイカが合成で減った / 消えたとき。
WATERMELON_CLEAR_BONUS = 25.0
# ダブルスイカ消去での成功終了 (最高点)。
WIN_BONUS = 200.0
# 後方互換エイリアス (旧名・到達ボーナス)。
DOUBLE_WATERMELON_BONUS = DOUBLE_REACH_BONUS
# ゲームオーバー。
DEATH_PENALTY = -20.0


def is_game_over(obs: Observation) -> bool:
    """頭頂が負けラインを超えているか。"""
    if not obs.fruits:
        return False
    crown = min(f.y - f.radius for f in obs.fruits)
    return crown < GAME_OVER_Y


def watermelon_count(obs: Observation) -> int:
    return sum(1 for f in obs.fruits if f.type == WATERMELON)


def cleared_double_watermelon(
    before: Observation,
    after: Observation,
    *,
    merges: int,
) -> bool:
    """2 個以上あったスイカが合成で減ったか (成功条件)。"""
    if merges <= 0:
        return False
    before_w = watermelon_count(before)
    after_w = watermelon_count(after)
    return before_w >= 2 and after_w < before_w


def _max_fruit_type(obs: Observation) -> int:
    if not obs.fruits:
        return -1
    return max(f.type for f in obs.fruits)


def step_reward(
    before: Observation,
    after: Observation,
    *,
    merges: int,
    done: bool,
    win: bool = False,
) -> float:
    """1 手分の報酬。

    - 生存と合成を基本報酬にする
    - 盤の最大段階の更新・スイカ増・ダブル到達・スイカ消去を加点
    - ダブル消去の成功終了で WIN_BONUS (最高点)
    - ゲームオーバーで大きく減点
    """
    if done and not win:
        return DEATH_PENALTY

    reward = STEP_REWARD
    if merges > 0:
        # 合成後の最大 type を粗く重みにする (無ければ held 想定で merges のみ)。
        grown = _max_fruit_type(after)
        weight = MERGE_WEIGHT * (1.0 + max(grown, 0) * 0.15)
        reward += merges * weight

    before_max = _max_fruit_type(before)
    after_max = _max_fruit_type(after)
    if after_max > before_max:
        reward += (after_max - before_max) * PROGRESS_WEIGHT

    before_w = watermelon_count(before)
    after_w = watermelon_count(after)
    if after_w > before_w:
        reward += (after_w - before_w) * WATERMELON_BONUS
    if before_w < 2 <= after_w:
        reward += DOUBLE_REACH_BONUS
    if merges > 0 and after_w < before_w:
        reward += (before_w - after_w) * WATERMELON_CLEAR_BONUS

    if win:
        reward += WIN_BONUS

    # 高い山は将来の死に近づくので小さく減点 (即死の手前)。
    if after.fruits:
        crown = min(f.y - f.radius for f in after.fruits)
        headroom = (crown - GAME_OVER_Y) / max(NORMALIZED_HEIGHT - GAME_OVER_Y, 1.0)
        if headroom < 0.25:
            reward -= (0.25 - headroom) * 2.0

    return float(reward)
