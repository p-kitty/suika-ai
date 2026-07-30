"""落とす列を決める。薄い bootstrap 方策 (RL の土台)。

具体手順 (押し込み・育成優先・連鎖隙間など) は持たない。
合成・危険高さ・埋め込み・薄い大小順・転がり事故防止だけ見る。
"""

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
# next 手の割引。
NEXT_DISCOUNT = 0.55
# 着地後の転がり。側面に乗ったら谷まで横へずらす。
SETTLE_STEP = 3.0
SETTLE_MAX_ITERS = 48
# 異種のほぼ中央真上への減点。
FOREIGN_CENTER_PENALTY = 140.0
# 大小逆転ペアの type 差あたり減点。
SIZE_ORDER_PAIR_WEIGHT = 28.0
# ideal 列からの平均距離あたり減点 (弱め。レイアウト強制にしない)。
SIZE_ORDER_IDEAL_WEIGHT = 0.2
# 自分より2段階以上大きい実どうしの隙間に詰める減点。
GAP_JUNK_PENALTY = 200.0
# ideal 列への弱い引力。
IDEAL_PULL = 0.25


def choose_x(obs: Observation) -> float:
    """観測から落とす列を返す。ready で held_type がある前提。"""
    if obs.held_type is None:
        raise ValueError("held_type が無い")

    held_r = _radius(obs.held_type)
    best_x = NORMALIZED_WIDTH / 2
    best_score = -math.inf

    for x in _candidates(obs.fruits, obs.held_type, held_r, extra_type=obs.next_type):
        x = clamp_drop_x(x, obs.held_type)
        score = _score(obs, x, held_r)
        if score > best_score:
            best_score = score
            best_x = x

    return best_x


def _candidates(
    fruits: tuple[Fruit, ...] | list[Fruit],
    drop_type: int,
    held_r: float,
    extra_type: int | None = None,
) -> list[float]:
    """均等刻みに、同種・近い実の上／横と ideal_x を足す。"""
    sign = _order_sign(fruits)
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    xs = {round(x / CANDIDATE_STEP) * CANDIDATE_STEP for x in _frange(lo, hi, CANDIDATE_STEP)}
    xs.add(_ideal_x(drop_type, sign))

    for fruit in fruits:
        if fruit.type < drop_type or fruit.type > drop_type + 2:
            continue
        xs.add(fruit.x)
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)

    if extra_type is not None:
        xs.add(_ideal_x(extra_type, sign))
        for fruit in fruits:
            if fruit.type != extra_type:
                continue
            xs.add(fruit.x)
            gap = held_r + fruit.radius
            xs.add(fruit.x - gap)
            xs.add(fruit.x + gap)

    return [x for x in xs if lo <= x <= hi]


def _score(obs: Observation, x: float, held_r: float) -> float:
    """held を落とした盤＋ next の仮想最善手を採点する。"""
    assert obs.held_type is not None
    before = list(obs.fruits)
    sign = _order_sign(before)
    land_x, land_y = _preview_land(before, obs.held_type, x, held_r)
    after, merges = simulate_drop(before, obs.held_type, x)

    score = _board_score(after, merges, land_y=land_y, sign=sign)
    score -= _foreign_center_penalty(before, x, land_x, land_y, obs.held_type, held_r)
    if merges == 0:
        score -= _wrong_side_roll_penalty(
            before, land_x, land_y, obs.held_type, held_r, sign
        )
        score -= _gap_junk_penalty(before, land_x, land_y, obs.held_type, held_r)
        score -= abs(x - _ideal_x(obs.held_type, sign)) * IDEAL_PULL
    score -= _coast_away_penalty(before, x, land_x, land_y, held_r)

    if obs.next_type is not None:
        score += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)

    return score


def _best_next_score(fruits: list[Fruit], next_type: int) -> float:
    """next を最善列に落としたときの盤面スコア。"""
    next_r = _radius(next_type)
    sign = _order_sign(fruits)
    best = -math.inf
    for nx in _candidates(fruits, next_type, next_r):
        nx = clamp_drop_x(nx, next_type)
        land_x, land_y = _preview_land(fruits, next_type, nx, next_r)
        after, merges = simulate_drop(fruits, next_type, nx)
        value = _board_score(after, merges, land_y=land_y, sign=sign)
        value -= _foreign_center_penalty(fruits, nx, land_x, land_y, next_type, next_r)
        if merges == 0:
            value -= _wrong_side_roll_penalty(
                fruits, land_x, land_y, next_type, next_r, sign
            )
            value -= _gap_junk_penalty(fruits, land_x, land_y, next_type, next_r)
            value -= abs(nx - _ideal_x(next_type, sign)) * IDEAL_PULL
        value -= _coast_away_penalty(fruits, nx, land_x, land_y, next_r)
        if value > best:
            best = value
    return 0.0 if best == -math.inf else best


def _board_score(
    fruits: list[Fruit],
    merges: int,
    *,
    land_y: float,
    sign: int = 1,
) -> float:
    """1 手分の盤面評価（合成・高さ・埋め込み・サイズ順）。"""
    score = 0.0
    score += 140.0 * merges
    if merges >= 2:
        score += 80.0 * (merges - 1)

    score += land_y * 0.22

    crown = _top_crown(fruits)
    score += crown * 0.8
    if crown < DANGER_Y:
        score -= (DANGER_Y - crown) * 4.0

    score -= 90.0 * _bury_penalty(fruits)
    score -= _size_order_penalty(fruits, sign)
    variance = _height_variance(fruits)
    if crown < DANGER_Y:
        variance *= 0.15
    score -= 1.2 * variance
    return score


def _foreign_center_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_x: float,
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
) -> float:
    """同種以外のほぼ中央真上に落とす減点。"""
    for fruit in fruits:
        if fruit.type == drop_type:
            continue
        if not _is_on_top(fruit, land_x, held_r, land_y):
            continue
        if abs(drop_x - fruit.x) <= fruit.radius * 0.25:
            return FOREIGN_CENTER_PENALTY
    return 0.0


def _is_on_top(support: Fruit, x: float, held_r: float, land_y: float) -> bool:
    """support のほぼ真上に着地しているか。"""
    if abs(x - support.x) > support.radius * 0.85:
        return False
    top = support.y - support.radius
    return abs((land_y + held_r) - top) <= MERGE_SLACK


def _wrong_side_roll_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
    sign: int,
) -> float:
    """転がって大きい実の大側床に落ちたときの減点。"""
    floor = NORMALIZED_HEIGHT - held_r
    if land_y < floor - 4.0:
        return 0.0

    penalty = 0.0
    for other in fruits:
        if other.type <= drop_type:
            continue
        if (land_x - other.x) * sign >= 0:
            continue
        if abs(land_x - other.x) > other.radius + held_r + MERGE_SLACK * 2:
            continue
        penalty += 180.0 + 40.0 * (other.type - drop_type)
    return penalty


def _coast_away_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_x: float,
    land_x: float,
    land_y: float,
    held_r: float,
) -> float:
    """接触で弾かれて落下列から大きく離れた着地を減点する。"""
    floor = NORMALIZED_HEIGHT - held_r
    drifted = abs(land_x - drop_x)
    if drifted < held_r * 2:
        return 0.0
    penalty = drifted * 1.4
    if land_y >= floor - 4.0 and drifted > NORMALIZED_WIDTH * 0.25:
        penalty += 120.0
    return penalty


def _gap_junk_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
) -> float:
    """自分より2段階以上大きい実どうしの間に小さい実を詰める減点。"""
    _ = land_y
    left_big: Fruit | None = None
    right_big: Fruit | None = None
    for fruit in fruits:
        if fruit.type <= drop_type:
            continue
        if fruit.x < land_x:
            if left_big is None or fruit.x > left_big.x:
                left_big = fruit
        elif fruit.x > land_x:
            if right_big is None or fruit.x < right_big.x:
                right_big = fruit
    if left_big is None or right_big is None:
        return 0.0

    if min(left_big.type, right_big.type) - drop_type < 2:
        return 0.0

    sep = right_big.x - left_big.x
    touch = left_big.radius + right_big.radius
    if sep > touch + held_r * 2.8 + MERGE_SLACK:
        return 0.0
    return GAP_JUNK_PENALTY


def _ideal_x(fruit_type: int, sign: int = 1) -> float:
    """sign=+1 なら大きいほど左。sign=-1 なら大きいほど右。"""
    base = NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))
    if sign < 0:
        return NORMALIZED_WIDTH - base
    return base


def _order_sign(fruits: list[Fruit] | tuple[Fruit, ...]) -> int:
    """盤面の大小の向き。+1=左大右小、-1=左小右大。"""
    if not fruits:
        return 1
    if len(fruits) == 1:
        fruit = fruits[0]
        if fruit.type >= 4 and fruit.x > NORMALIZED_WIDTH * 0.55:
            return -1
        return 1

    votes = 0.0
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if a.type == b.type:
                continue
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            weight = float(abs(a.type - b.type)) * (1.0 + 0.15 * max(a.type, b.type))
            if left.type > right.type:
                votes += weight
            else:
                votes -= weight

    if abs(votes) < 1.0:
        biggest = max(fruits, key=lambda f: (f.type, f.radius))
        return -1 if biggest.x > NORMALIZED_WIDTH * 0.5 else 1
    return 1 if votes > 0 else -1


def _size_order_penalty(fruits: list[Fruit], sign: int = 1) -> float:
    """左右の大小が逆転しているペアを減点。絶対 ideal より相対順を見る。"""
    if not fruits:
        return 0.0
    penalty = 0.0
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            if sign > 0 and left.type < right.type:
                penalty += (right.type - left.type) * SIZE_ORDER_PAIR_WEIGHT
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * SIZE_ORDER_PAIR_WEIGHT
    penalty += (
        sum(abs(f.x - _ideal_x(f.type, sign)) for f in fruits)
        / len(fruits)
        * SIZE_ORDER_IDEAL_WEIGHT
    )
    return penalty


def simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int]:
    """列 x に落としたあとの盤面と合成回数。sim / 学習用。"""
    placed = list(fruits)
    placed, dropped = _place(placed, fruit_type, x)
    return _resolve_merges(placed, active={dropped})


def _after_drop(obs: Observation, x: float) -> tuple[list[Fruit], int]:
    """テスト用。held を列 x に落としたあとの盤面と合成回数。"""
    assert obs.held_type is not None
    return simulate_drop(obs.fruits, obs.held_type, x)


def _preview_land(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
    held_r: float,
) -> tuple[float, float]:
    """落下列 x から、転がり後の着地 (x, y)。"""
    x = _settle_x(fruits, x, held_r, allow_coast=True)
    return x, _land_y(fruits, x, held_r)


def _place(
    fruits: list[Fruit],
    fruit_type: int,
    x: float,
    *,
    allow_coast: bool = True,
) -> tuple[list[Fruit], int]:
    """列 x にフルーツを着地させて追加する。追加した index も返す。"""
    r = _radius(fruit_type)
    x = _settle_x(fruits, x, r, allow_coast=allow_coast)
    y = _land_y(fruits, x, r)
    fruits.append(Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=100.0))
    return fruits, len(fruits) - 1


def _settle_x(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
    *,
    allow_coast: bool = True,
) -> float:
    """円の側面に乗ったら谷・床まで転がし、床では惰性で壁／他実まで滑る。"""
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    x = max(lo, min(hi, x))
    floor = NORMALIZED_HEIGHT - held_r
    coast_dir = 0.0

    for _ in range(SETTLE_MAX_ITERS):
        y = _land_y(fruits, x, held_r)
        if y >= floor - 1.0:
            if coast_dir == 0.0 or not allow_coast:
                return x
            return _coast_on_floor(fruits, x, held_r, coast_dir)

        push = 0.0
        for fruit in fruits:
            dx = x - fruit.x
            gap = fruit.radius + held_r
            if abs(dx) >= gap - 1e-6:
                continue
            dy = math.sqrt(max(0.0, gap * gap - dx * dx))
            if abs((fruit.y - dy) - y) > 2.0:
                continue
            push += dx

        if abs(push) < 0.75:
            return x

        coast_dir = math.copysign(1.0, push)
        nxt = max(lo, min(hi, x + coast_dir * SETTLE_STEP))
        nxt_y = _land_y(fruits, nxt, held_r)
        if nxt_y < y - 0.5:
            return x
        if abs(nxt - x) < 1e-6:
            return x
        x = nxt
    return x


def _coast_on_floor(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
    direction: float,
) -> float:
    """斜面から床へ落ちたあと、その向きに壁か他実の接触まで滑る。"""
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    floor = NORMALIZED_HEIGHT - held_r
    direction = math.copysign(1.0, direction)
    max_iters = int(NORMALIZED_WIDTH / SETTLE_STEP) + 5

    for _ in range(max_iters):
        nxt = max(lo, min(hi, x + direction * SETTLE_STEP))
        if abs(nxt - x) < 1e-6:
            return x
        nxt_y = _land_y(fruits, nxt, held_r)
        if nxt_y < floor - 1.0:
            return x
        for fruit in fruits:
            limit = fruit.radius + held_r
            if abs(nxt - fruit.x) < limit - 0.5:
                if direction > 0:
                    return max(lo, min(hi, fruit.x - limit))
                return max(lo, min(hi, fruit.x + limit))
        x = nxt
    return x


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

        fruits, new_i = _place(fruits, new_type, mid_x, allow_coast=False)
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
    """合成候補を異種で埋める度合い。"""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type == under.type:
                continue
            if over.type < under.type:
                continue
            if over.y >= under.y:
                continue
            if abs(over.x - under.x) > under.radius * 0.9:
                continue
            gap = (under.y - under.radius) - (over.y + over.radius)
            if -MERGE_SLACK <= gap <= under.radius * 0.6:
                siblings = sum(1 for f in fruits if f.type == under.type and f is not under)
                if siblings >= 1:
                    penalty += 1.0
                else:
                    penalty += 0.35
    return penalty


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
    """列 x に落としたときの中心 y。床か、円どうしが接する位置。"""
    best = float(NORMALIZED_HEIGHT) - held_r
    for fruit in fruits:
        dx = abs(fruit.x - x)
        gap = fruit.radius + held_r
        if dx >= gap:
            continue
        dy = math.sqrt(gap * gap - dx * dx)
        best = min(best, fruit.y - dy)
    return best


def _radius(fruit_type: int) -> float:
    return fruit_radius_ratios()[fruit_type] * NORMALIZED_WIDTH


def _frange(start: float, stop: float, step: float):
    x = start
    while x <= stop + 1e-6:
        yield x
        x += step
