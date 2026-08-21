"""学習用の報酬。本家スイカゲームと同じ合成点のみ。"""

from __future__ import annotations

from collections.abc import Sequence

from .observe import Observation
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

WATERMELON = MAX_FRUIT_TYPE
# この y より上に頭頂が出たら負け (y は下向き)。盤面が壁の内側基準に
# なった分だけ、旧基準の 40.0 を座標変換してある。
GAME_OVER_Y = 14.9

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


def is_lost(fruits: Sequence[Fruit]) -> bool:
    """頭頂が負けラインを超えているか。落下後の盤をそのまま渡せる形。

    方策が候補を「その手で死ぬか」で振り分けるのに使うので、
    Observation ではなく実の列で受ける (`policy.choose_x`)。
    """
    if not fruits:
        return False
    crown = min(f.y - f.radius for f in fruits)
    return crown < GAME_OVER_Y


def is_game_over(obs: Observation) -> bool:
    return is_lost(obs.fruits)


# 角の判定の許容。スイカ (半径 112) が壁と床に付いたときの角の隙間には
# さくらんぼ (半径 14.2) がそのまま収まり、スイカは動かない。いちごなら
# スイカを壁から 4.9、ぶどうなら 23.2 押し出す (どちらも壁・床・スイカに
# 接する円の位置から出る)。その間を取り、いちごまでは角として数える。
CORNER_SLACK = 12.0


def is_corner_watermelon(fruits: Sequence[Fruit]) -> bool:
    """壁と床に密着したスイカがあるか。

    角スイカは目標形 (→NOTES「いまの進め方」)。到達したかを外から見られるよう、
    盤の実だけから判定する。角に小実が挟まっていても、押し出された量が
    `CORNER_SLACK` に収まっていれば角として数える。
    """
    for fruit in fruits:
        if fruit.type != WATERMELON:
            continue
        if fruit.y + fruit.radius < NORMALIZED_HEIGHT - CORNER_SLACK:
            continue
        wall_gap = min(fruit.x - fruit.radius, NORMALIZED_WIDTH - fruit.x - fruit.radius)
        if wall_gap <= CORNER_SLACK:
            return True
    return False


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
