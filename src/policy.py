"""落とす列を決める。仮想落下＋合成のヒューリスティック。"""

from __future__ import annotations

import math
import statistics

from .observe import Observation, clamp_drop_x
from .vision.classify import fruit_radius_ratios
from .vision.colors import FRUIT_NAMES
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# 候補列の刻み (正規化座標)。
CANDIDATE_STEP = 8.0
# 合成できそうな接触の許容 (中心距離と半径和の差)。候補評価用。
MERGE_SLACK = 18.0
# 仮想合成の接触。観測盤は静止前提なので緩めすぎない。
CONTACT_SLACK = 2.0
# この y より上に頭が出ると危険 (盤面上辺寄り)。
DANGER_Y = 90.0
# 平坦さ評価用の列幅。
FLAT_BIN = 40.0
# スイカ。これ以上は合成しない。
MAX_FRUIT_TYPE = len(FRUIT_NAMES) - 1


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
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)

    if obs.next_type is not None:
        for fruit in obs.fruits:
            if fruit.type != obs.next_type:
                continue
            xs.add(fruit.x)
            gap = held_r + fruit.radius
            xs.add(fruit.x - gap)
            xs.add(fruit.x + gap)

    return [x for x in xs if lo <= x <= hi]


def _score(obs: Observation, x: float, held_r: float) -> float:
    """着手後の盤面を見て採点する。"""
    before = list(obs.fruits)
    after, merges = _after_drop(obs, x)
    land_y = _land_y(before, x, held_r)

    score = 0.0

    score += 140.0 * merges
    if merges >= 2:
        score += 80.0 * (merges - 1)

    score += land_y * 0.35

    crown = _top_crown(after)
    score += crown * 0.8
    if crown < DANGER_Y:
        score -= (DANGER_Y - crown) * 4.0

    score -= 90.0 * _bury_penalty(after)

    if obs.next_type is not None:
        score += 70.0 * _next_setup_at(before, after, x, obs.next_type, merges)

    score -= 1.2 * _height_variance(after)

    if merges == 0 and not _column_fruits(before, x, held_r):
        score += 8.0

    score -= abs(x - NORMALIZED_WIDTH / 2) * 0.05

    return score


def _after_drop(obs: Observation, x: float) -> tuple[list[Fruit], int]:
    """列 x に落として合成を解決した盤面と合成回数。"""
    assert obs.held_type is not None
    fruits = list(obs.fruits)
    fruits, dropped = _place(fruits, obs.held_type, x)
    return _resolve_merges(fruits, active={dropped})


def _place(fruits: list[Fruit], fruit_type: int, x: float) -> tuple[list[Fruit], int]:
    """列 x にフルーツを着地させて追加する。追加した index も返す。"""
    r = _radius(fruit_type)
    y = _land_y(fruits, x, r)
    fruits.append(Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=100.0))
    return fruits, len(fruits) - 1


def _resolve_merges(fruits: list[Fruit], active: set[int]) -> tuple[list[Fruit], int]:
    """落とした実から始まる同種接触だけを合成する。観測盤は静止前提。"""
    fruits = list(fruits)
    merges = 0
    for _ in range(64):
        pair = _find_merge_pair(fruits, active)
        if pair is None:
            break
        i, j = pair
        a, b = fruits[i], fruits[j]
        new_type = a.type + 1
        mid_x = (a.x + b.x) / 2
        for idx in sorted((i, j), reverse=True):
            fruits.pop(idx)

        if new_type > MAX_FRUIT_TYPE:
            active = set()
            merges += 1
            continue

        fruits, new_i = _place(fruits, new_type, mid_x)
        active = {new_i}
        merges += 1

    return fruits, merges


def _find_merge_pair(fruits: list[Fruit], active: set[int]) -> tuple[int, int] | None:
    """active 側と接触している同種ペア。"""
    for i in sorted(active):
        if i < 0 or i >= len(fruits):
            continue
        a = fruits[i]
        for j, b in enumerate(fruits):
            if j == i or b.type != a.type:
                continue
            if _touching(a, b):
                return (i, j) if i < j else (j, i)
    return None


def _touching(a: Fruit, b: Fruit) -> bool:
    dist = math.hypot(a.x - b.x, a.y - b.y)
    return dist <= a.radius + b.radius + CONTACT_SLACK


def _top_crown(fruits: list[Fruit]) -> float:
    """一番上の頭頂 y。空なら床。"""
    if not fruits:
        return float(NORMALIZED_HEIGHT)
    return min(f.y - f.radius for f in fruits)


def _bury_penalty(fruits: list[Fruit]) -> float:
    """同種の直上に異種が乗っている度合い (0〜)。"""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type == under.type:
                continue
            if abs(over.x - under.x) > under.radius * 0.9:
                continue
            gap = (over.y - over.radius) - (under.y + under.radius)
            if -MERGE_SLACK <= gap <= under.radius * 0.6:
                siblings = sum(1 for f in fruits if f.type == under.type and f is not under)
                if siblings >= 1:
                    penalty += 1.0
                else:
                    penalty += 0.35
    return penalty


def _next_setup_at(
    before: list[Fruit],
    after: list[Fruit],
    x: float,
    next_type: int,
    merges: int,
) -> float:
    """今の落下列が next 同種の近く／露出を壊さないほど高い (0〜1+)。"""
    targets = [f for f in before if f.type == next_type]
    if not targets:
        return 0.0

    next_r = _radius(next_type)
    best = 0.0
    for fruit in targets:
        dist = abs(x - fruit.x)
        reach = fruit.radius + next_r + MERGE_SLACK
        if dist <= reach:
            proximity = 1.0 - dist / max(reach, 1.0)
        elif dist <= reach * 2.5:
            proximity = 0.35 * (1.0 - (dist - reach) / (reach * 1.5))
        else:
            proximity = 0.0

        buried = any(
            abs(f.x - fruit.x) <= fruit.radius * 0.9 and f.type != next_type and f.y < fruit.y
            for f in after
        )
        if buried:
            proximity *= 0.15
        best = max(best, proximity)

    if merges > 0:
        return best * 0.35
    return best


def _height_variance(fruits: list[Fruit]) -> float:
    """列ビンごとの頭頂のばらつき。空なら 0。"""
    bins: dict[int, float] = {}
    for fruit in fruits:
        key = int(fruit.x // FLAT_BIN)
        top = fruit.y - fruit.radius
        bins[key] = min(bins.get(key, float(NORMALIZED_HEIGHT)), top)
    if len(bins) < 2:
        return 0.0
    return float(statistics.pstdev(list(bins.values())))


def _land_y(fruits: tuple[Fruit, ...] | list[Fruit], x: float, held_r: float) -> float:
    """列 x に落としたときの中心 y。床か、重なるフルーツの頭頂の上。"""
    top = float(NORMALIZED_HEIGHT)
    for fruit in fruits:
        if abs(fruit.x - x) > fruit.radius + held_r:
            continue
        top = min(top, fruit.y - fruit.radius)
    return top - held_r


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
