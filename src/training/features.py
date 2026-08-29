"""落下後の盤を固定長の特徴にする。価値関数の入力用。

`encode.py` と役が違う。あちらは**落とす前**の盤を実の羅列で渡すもので、
候補ごとの着地結果が入っていない。1 隠れ層 MLP がそこから教師の判断を
真似るのは、lr / epoch / hidden / soft-hard を総当たりしても match 30% で
頭打ちになった（NOTES「BC が match 60-70% に届かない」）。こちらは候補を
実際に落としたあとの盤を数える側で、教師と同じ情報を学習側に渡すためにある。

**項ごとに分けて渡す。合計しない。** `board_penalties` の総和や `eval` を
1 本の数として渡すと、学習側は重み配分を学び直せない。重みが乗ったままなのは
構わない（線形の係数に吸収される）が、**足してしまうと分離できない**。
`size_order` は性質の違う 2 規則の和なので、ここでも割って渡す
（NOTES「合成項をサブ項へ割る」）。

角スイカ（目標形）に効く量は幾何側に置いてある。最大実の壁・床の隙間は
NOTES「調査済み: 角スイカが出ない原因」が 20 局のトレースで測った分かれ目
そのもので、`penalties.py` の減点では表せないと分かっている
（`_corner_lift_penalty` は帯の外へ 0.0%。→NOTES「効かなかった案」）。
"""

from __future__ import annotations

import numpy as np

from .. import penalties as pen
from ..reward import CORNER_SLACK, GAME_OVER_Y, WATERMELON
from ..vision.colors import MAX_FRUIT_TYPE
from ..vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from ..vision.state import Fruit

# 特徴の並び。debug と後段の重み読みのために名前を持たせる。
FEATURE_NAMES: tuple[str, ...] = (
    "fruit_count",
    "crown_margin",
    "max_type",
    "units",
    "bury_pair",
    "bury_lone",
    "perch",
    "pit",
    "excess_same",
    "size_order_pair",
    "size_order_ideal",
    "corner_pocket",
    "big_wall_gap",
    "big_floor_gap",
    "big_cornered",
    "melon_count",
    "watermelon_count",
    "mean_height",
    "sign",
)
FEATURE_DIM = len(FEATURE_NAMES)

# 割り算の分母。特徴を 0〜1 付近に収めるだけの目安で、意味は無い。
_COUNT_SCALE = 32.0
_UNIT_SCALE = 1024.0


def _size_order_parts(fruits: list[Fruit], sign: int) -> tuple[float, float]:
    """`_size_order_penalty` を (ペア分, ideal 乖離分) に割る。

    ideal 側だけ本家と同じ式で引き直し、残りをペア分とする。ペアの走査は
    `_size_order_exempt` の判定を含んで込み入っているので、写して持たない
    （写すと本家が変わったときに黙ってずれる）。
    """
    total = pen._size_order_penalty(fruits, sign)
    exempt = [pen._size_order_exempt(f, fruits) for f in fruits]
    open_fruits = [f for f, skip in zip(fruits, exempt) if not skip]
    ideal = 0.0
    if open_fruits:
        ideal = (
            sum(abs(f.x - pen.ideal_x(f.type, sign)) for f in open_fruits)
            / len(open_fruits)
            * pen.SIZE_ORDER_IDEAL_WEIGHT
        )
    return total - ideal, ideal


def board_features(fruits: list[Fruit] | tuple[Fruit, ...], *, sign: int) -> np.ndarray:
    """落下後の盤 -> float32 ベクトル (FEATURE_DIM,)。

    sign は盤面の大小の向き（`policy._order_sign`）。角ポケット・大小順・
    壁の隙間はどちら側を大側と見るかで値が変わるので、呼び元が渡す。
    """
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    board = list(fruits)
    if not board:
        # 空盤。crown_margin だけは「死から最も遠い」を表す 1.0 にする。
        out[FEATURE_NAMES.index("crown_margin")] = 1.0
        out[FEATURE_NAMES.index("sign")] = float(sign)
        return out

    max_t = max(f.type for f in board)
    biggest = max(board, key=lambda f: (f.type, -f.y))
    crown = min(f.y - f.radius for f in board)
    pair, lone = pen._bury_counts(board)
    so_pair, so_ideal = _size_order_parts(board, sign)

    values = {
        "fruit_count": len(board) / _COUNT_SCALE,
        # 負けラインまでの余裕。負なら死んでいる。
        "crown_margin": (crown - GAME_OVER_Y) / NORMALIZED_HEIGHT,
        "max_type": max_t / MAX_FRUIT_TYPE,
        # 合体で保存される量＝盤に積んだ材料の総和（NOTES「材料の計算」）。
        "units": sum(2.0**f.type for f in board) / _UNIT_SCALE,
        "bury_pair": pair,
        "bury_lone": lone,
        "perch": pen._perch_penalty(board),
        "pit": pen._pit_penalty(board),
        "excess_same": pen._excess_same_penalty(board),
        "size_order_pair": so_pair,
        "size_order_ideal": so_ideal,
        "corner_pocket": pen._corner_pocket_penalty(board, sign),
        # 角スイカの分かれ目。壁側は大側だけ見る（`pen.wall_gap` が sign 依存）。
        "big_wall_gap": pen.wall_gap(biggest, sign) / NORMALIZED_WIDTH,
        "big_floor_gap": (NORMALIZED_HEIGHT - (biggest.y + biggest.radius))
        / NORMALIZED_HEIGHT,
        "big_cornered": 0.0,
        "melon_count": sum(1 for f in board if f.type == WATERMELON - 1),
        "watermelon_count": sum(1 for f in board if f.type == WATERMELON),
        "mean_height": sum(NORMALIZED_HEIGHT - f.y for f in board)
        / len(board)
        / NORMALIZED_HEIGHT,
        "sign": float(sign),
    }
    # 壁と床の両方に密着しているか。角スイカ判定 (`is_corner_watermelon`) と
    # 同じ許容を使う。型は問わない（メロン段階の角も進捗として見せる）。
    values["big_cornered"] = float(
        pen.wall_gap(biggest, sign) <= CORNER_SLACK
        and NORMALIZED_HEIGHT - (biggest.y + biggest.radius) <= CORNER_SLACK
    )

    for i, name in enumerate(FEATURE_NAMES):
        out[i] = values[name]
    return out
