"""落とす列を決める。最初はヒューリスティック。"""

from __future__ import annotations

import math

from .observe import Observation, clamp_drop_x
from .vision.classify import fruit_radius_ratios
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# 候補列の刻み (正規化座標)。
CANDIDATE_STEP = 8.0
# 合成できそうな接触の許容 (中心距離と半径和の差)。
MERGE_SLACK = 18.0
# この y より上に頭が出ると危険 (盤面上辺寄り)。
DANGER_Y = 90.0


def choose_x(obs: Observation) -> float:
    """観測から落とす列を返す。ready で held_type がある前提。"""
    if obs.held_type is None:
        raise ValueError("held_type が無い")

    held_r = _radius(obs.held_type)
    best_x = NORMALIZED_WIDTH / 2
    best_score = -math.inf

    for x in _candidates(obs, held_r):
        x = clamp_drop_x(x, obs.held_type)
        score = _score(obs, x, held_r)
        if score > best_score:
            best_score = score
            best_x = x

    return best_x


def _candidates(obs: Observation, held_r: float) -> list[float]:
    """均等刻みに、同種の上／横を足す。"""
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    xs = {round(x / CANDIDATE_STEP) * CANDIDATE_STEP for x in _frange(lo, hi, CANDIDATE_STEP)}

    for fruit in obs.fruits:
        if fruit.type != obs.held_type:
            continue
        xs.add(fruit.x)
        # 横に付けて合成を狙う。
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)

    if obs.next_type is not None:
        for fruit in obs.fruits:
            if fruit.type != obs.next_type:
                continue
            xs.add(fruit.x)

    return [x for x in xs if lo <= x <= hi]


def _score(obs: Observation, x: float, held_r: float) -> float:
    land_y = _land_y(obs.fruits, x, held_r)
    score = 0.0

    # 低い位置に落ちるほどよい (y が大きい)。
    score += land_y

    # 同種に接触しそうなら大きく加点。
    score += 120.0 * _merge_chance(obs.fruits, obs.held_type, x, land_y, held_r)

    # 次のフルーツと同種の近くは、盤面を崩しすぎない程度に少し優遇。
    if obs.next_type is not None:
        score += 25.0 * _merge_chance(obs.fruits, obs.next_type, x, land_y, held_r)

    # 頭が上辺に近づくほど減点。
    crown = land_y - held_r
    if crown < DANGER_Y:
        score -= (DANGER_Y - crown) * 3.0

    # 空の列への落としをわずかに優遇 (序盤のばらまき)。
    if not _column_fruits(obs.fruits, x, held_r):
        score += 8.0

    # 同点なら中央寄り。空盤で端だけ選ばないようにする。
    score -= abs(x - NORMALIZED_WIDTH / 2) * 0.05

    return score


def _land_y(fruits: tuple[Fruit, ...] | list[Fruit], x: float, held_r: float) -> float:
    """列 x に落としたときの中心 y。床か、重なるフルーツの頭頂の上。"""
    top = float(NORMALIZED_HEIGHT)
    for fruit in fruits:
        if abs(fruit.x - x) > fruit.radius + held_r:
            continue
        top = min(top, fruit.y - fruit.radius)
    return top - held_r


def _merge_chance(
    fruits: tuple[Fruit, ...] | list[Fruit],
    fruit_type: int | None,
    x: float,
    land_y: float,
    held_r: float,
) -> float:
    """0〜1。落ちた位置が同種に触れそうなら高い。"""
    if fruit_type is None:
        return 0.0

    best = 0.0
    for fruit in fruits:
        if fruit.type != fruit_type:
            continue
        dist = math.hypot(x - fruit.x, land_y - fruit.y)
        expected = held_r + fruit.radius
        gap = abs(dist - expected)
        if gap >= MERGE_SLACK:
            # 真上に落とす場合も合成対象になりやすい。
            if abs(x - fruit.x) <= max(held_r, fruit.radius) * 0.85:
                best = max(best, 0.55)
            continue
        best = max(best, 1.0 - gap / MERGE_SLACK)
    return best


def _column_fruits(
    fruits: tuple[Fruit, ...] | list[Fruit],
    x: float,
    held_r: float,
) -> list[Fruit]:
    return [f for f in fruits if abs(f.x - x) <= f.radius + held_r]


def _radius(fruit_type: int) -> float:
    return fruit_radius_ratios()[fruit_type] * NORMALIZED_WIDTH


def _frange(start: float, stop: float, step: float):
    x = start
    while x <= stop + 1e-6:
        yield x
        x += step
