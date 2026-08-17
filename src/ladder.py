"""梯子 (角の大実を階段で発火させる形) の検出。

桃(7) を角に置き、内側の隣に梨(6)。その 2 つの肩にリンゴ(5)・オレンジ(4)。
最後にオレンジを落とすと 4→5→6→7 と連鎖して角の桃がパインになる。
角パイン以降も同じで、階段は常にツモれる最大 (orange) まで降りる。

**いまは検出だけで、手選びには使っていない。** `policy.choose_x` からは
呼ばれず、参照しているのは tests/test_policy.py だけ。実測で分かっていること:

- 発火は誘導不要。梯子が組み上がっていれば choose_x は x 全振りの最良と同点を取る
- 組み上がる盤が出ない (実測 720 盤中 4 段揃いは 12 回)。手を入れるならここ
- 床が埋まっていないと形が保たない。梨が楔で押し出されて自壊し、
  どこに落としても 1 段ぶん (15 点) しか取れない。床の詰まりはゲート条件

詳しくは NOTES.md「進行中: 床埋め後の大ツモ・梯子」。
"""

from __future__ import annotations

from .penalties import MERGE_SLACK, is_wall_anchored, wall_gap
from .vision.classify import fruit_radius
from .vision.colors import SPAWN_MAX_TYPE
from .vision.state import Fruit

# 土台として認める最小の型 (桃)。
MIN_ANCHOR_TYPE = 7
# 梯子の最下段。
BASE_TYPE = SPAWN_MAX_TYPE
# その下 (dekopon)。デコポン 2 個でオレンジ段を作れるので、
# held/next が両方デコポンのときだけ段として認める。
FEED_TYPE = SPAWN_MAX_TYPE - 1


def find_anchor(
    fruits: list[Fruit] | tuple[Fruit, ...],
    sign: int,
) -> Fruit | None:
    """梯子の土台。大側の壁に付いた最大実。桃未満・壁から離れていれば None。"""
    if not fruits:
        return None
    max_t = max(fruit.type for fruit in fruits)
    if max_t < MIN_ANCHOR_TYPE:
        return None
    best: Fruit | None = None
    for fruit in fruits:
        if fruit.type != max_t or not is_wall_anchored(fruit, sign):
            continue
        if best is None or wall_gap(fruit, sign) < wall_gap(best, sign):
            best = fruit
    return best


def _window(anchor: Fruit, sign: int) -> tuple[float, float]:
    """梯子が占める横帯。土台の中心すこし外側から、内側は梨 2 個ぶん。"""
    inner = anchor.radius + fruit_radius(anchor.type - 1) * 2.0 + MERGE_SLACK
    outer = anchor.radius * 0.5
    if sign > 0:
        return anchor.x - outer, anchor.x + inner
    return anchor.x - inner, anchor.x + outer


def _beside_anchor(anchor: Fruit, x: float, sign: int) -> bool:
    """土台の真上ではなく内側の隣か。

    ひとつ小さい段 (桃に対する梨) は隣に並べる。真上に積むと崩れる形になる。
    """
    return (x - anchor.x) * sign > anchor.radius * 0.5


def rungs(
    fruits: list[Fruit] | tuple[Fruit, ...],
    anchor: Fruit,
    sign: int,
) -> dict[int, Fruit]:
    """土台から下へ連続して埋まっている段。途切れたらそこで終わり。

    段は横帯の中にあり、ひとつ上の段より下がっていないものを壁側から取る。
    梨は桃の隣 (ほぼ同じ y) なので、床の半径差ぶんは許す。
    """
    lo, hi = _window(anchor, sign)
    found = {anchor.type: anchor}
    above = anchor
    for want in range(anchor.type - 1, FEED_TYPE - 1, -1):
        best: Fruit | None = None
        for fruit in fruits:
            if fruit.type != want or fruit is anchor:
                continue
            if not lo <= fruit.x <= hi:
                continue
            if fruit.y > above.y + above.radius:
                continue
            if want == anchor.type - 1 and not _beside_anchor(anchor, fruit.x, sign):
                continue
            if best is None or wall_gap(fruit, sign) < wall_gap(best, sign):
                best = fruit
        if best is None:
            break
        found[want] = best
        above = best
    return found
