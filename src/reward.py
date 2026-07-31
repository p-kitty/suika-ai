"""学習用の報酬。本家スイカゲームと同じ合成点のみ。"""

from __future__ import annotations

from collections.abc import Sequence

from .observe import Observation
from .vision.colors import MAX_FRUIT_TYPE

WATERMELON = MAX_FRUIT_TYPE
# この y より上に頭頂が出たら負け (y は下向き)。
GAME_OVER_Y = 40.0

# 合成でその段階の実ができたときの点数 (index = できた type)。
# cherry は落下のみで合成では生まれないので 0。
# ダブルスイカ消去は CREATE_SCORE 外の CLEAR_SCORE。
CREATE_SCORE: tuple[int, ...] = (
    0,   # cherry
    1,   # straw
    3,   # grape
    6,   # dekopon
    10,  # orange
    15,  # apple
    21,  # pear
    28,  # peach
    36,  # pineapple
    45,  # melon
    55,  # watermelon
)
# スイカ同士の合成で消えたとき。
CLEAR_SCORE = 65


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


def merge_points(source_type: int) -> int:
    """同種 source_type 同士を合成したときの本家点数。"""
    if source_type >= WATERMELON:
        return CLEAR_SCORE
    created = source_type + 1
    if 0 <= created < len(CREATE_SCORE):
        return CREATE_SCORE[created]
    return 0


def merge_score(merge_types: Sequence[int] = ()) -> float:
    """1 手分の本家点 = その手の合成点合計。減点・生存加点なし。"""
    return float(sum(merge_points(t) for t in merge_types))
