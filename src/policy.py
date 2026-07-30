"""落とす列を決める。サイズ順＋ held/next のヒューリスティック。"""

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
# 大小の間に中間段階の列を潰したときの、不足 px あたり減点。
CHAIN_SPACING_WEIGHT = 2.0


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
    """均等刻みに、同種・一段大きい実の上／横と ideal_x を足す。"""
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    xs = {round(x / CANDIDATE_STEP) * CANDIDATE_STEP for x in _frange(lo, hi, CANDIDATE_STEP)}
    xs.add(_ideal_x(drop_type))

    for fruit in fruits:
        # 同種、または少し大きい実の上／横 (オレンジ→リンゴなど)。
        if fruit.type < drop_type or fruit.type > drop_type + 2:
            continue
        xs.add(fruit.x)
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)

    if extra_type is not None:
        xs.add(_ideal_x(extra_type))
        for fruit in fruits:
            if fruit.type != extra_type:
                continue
            xs.add(fruit.x)
            gap = held_r + fruit.radius
            xs.add(fruit.x - gap)
            xs.add(fruit.x + gap)

    # 右の小さい実／左の大きい実との間に、中間段階の列を残す位置。
    for fruit in fruits:
        if fruit.type < drop_type:
            xs.add(fruit.x - _chain_center_gap(drop_type, fruit.type))
        elif fruit.type > drop_type:
            xs.add(fruit.x + _chain_center_gap(fruit.type, drop_type))

    return [x for x in xs if lo <= x <= hi]


def _score(obs: Observation, x: float, held_r: float) -> float:
    """held を落とした盤＋ next の仮想最善手を採点する。"""
    assert obs.held_type is not None
    before = list(obs.fruits)
    after, merges = _simulate_drop(before, obs.held_type, x)
    land_y = _land_y(before, x, held_r)
    cleared_wedge = _clears_wedged(before, x, obs.held_type, held_r, merges)
    grow_target = _growth_target_type(obs.held_type, obs.next_type)

    score = _board_score(after, merges, land_y=land_y)
    score += _wedged_priority(before, obs.held_type, cleared_wedge)
    score += _larger_neighbor_bonus(
        before, x, obs.held_type, held_r, land_y, grow_target=grow_target
    )
    # 挟まった同種を合成する手では、大きい実への寄りを強制しない。
    if not cleared_wedge:
        score -= _ignored_larger_penalty(before, x, obs.held_type, held_r, land_y)

    # 合成が無いときは、一段大きい実の「並ぶ側」寄り。
    # 床に並べるときだけ、中間段階の列潰しを減点する (積み重ねの x は対象外)。
    if merges == 0:
        score -= abs(x - _anchor_x(obs.held_type, before, held_r)) * 0.45
        floor = NORMALIZED_HEIGHT - held_r
        if land_y >= floor - 4.0:
            score -= _chain_spacing_penalty(before, x, obs.held_type)
        if not _column_fruits(before, x, held_r):
            score += 3.0

    if obs.next_type is not None:
        score += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)

    return score


def _growth_target_type(held_type: int, next_type: int | None) -> int | None:
    """held と next が同種なら、二個で一段大きい実を育てられる。その育成対象。"""
    if next_type is None or next_type != held_type:
        return None
    target = held_type + 1
    if target > MAX_FRUIT_TYPE:
        return None
    return target


def _best_next_score(fruits: list[Fruit], next_type: int) -> float:
    """next を最善列に落としたときの盤面スコア。"""
    next_r = _radius(next_type)
    best = -math.inf
    for nx in _candidates(fruits, next_type, next_r):
        nx = clamp_drop_x(nx, next_type)
        after, merges = _simulate_drop(fruits, next_type, nx)
        land_y = _land_y(fruits, nx, next_r)
        cleared_wedge = _clears_wedged(fruits, nx, next_type, next_r, merges)
        value = _board_score(after, merges, land_y=land_y)
        value += _wedged_priority(fruits, next_type, cleared_wedge)
        value += _larger_neighbor_bonus(fruits, nx, next_type, next_r, land_y)
        if not cleared_wedge:
            value -= _ignored_larger_penalty(fruits, nx, next_type, next_r, land_y)
        if merges == 0:
            value -= abs(nx - _anchor_x(next_type, fruits, next_r)) * 0.45
            floor = NORMALIZED_HEIGHT - next_r
            if land_y >= floor - 4.0:
                value -= _chain_spacing_penalty(fruits, nx, next_type)
        if value > best:
            best = value
    return 0.0 if best == -math.inf else best


def _chain_center_gap(left_type: int, right_type: int) -> float:
    """左(大)と右(小)の間に、中間段階を全部並べるときの中心距離。"""
    gap = _radius(left_type) + _radius(right_type)
    for mid in range(right_type + 1, left_type):
        gap += 2.0 * _radius(mid)
    return gap


def _chain_spacing_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
) -> float:
    """大きい順の列で、中間段階の隙間を潰す置きを減点する。

    例: 右端にイチゴがあるのにオレンジをすぐ左へ置くと、デコポン・グレープの
    並ぶ場所が無くなる。
    """
    penalty = 0.0
    for other in fruits:
        if other.type < drop_type and other.x > x:
            need = _chain_center_gap(drop_type, other.type)
            for mid in range(other.type + 1, drop_type):
                if any(x < f.x < other.x and f.type == mid for f in fruits):
                    need -= 2.0 * _radius(mid)
            have = other.x - x
            if have < need:
                penalty += (need - have) * CHAIN_SPACING_WEIGHT
        elif other.type > drop_type and other.x < x:
            need = _chain_center_gap(other.type, drop_type)
            for mid in range(drop_type + 1, other.type):
                if any(other.x < f.x < x and f.type == mid for f in fruits):
                    need -= 2.0 * _radius(mid)
            have = x - other.x
            if have < need:
                penalty += (need - have) * CHAIN_SPACING_WEIGHT
    return penalty


def _is_wedged(fruit: Fruit, fruits: list[Fruit] | tuple[Fruit, ...]) -> bool:
    """左右に自分より大きい実が近接して挟まっている。"""
    left_big: Fruit | None = None
    right_big: Fruit | None = None
    for other in fruits:
        if other is fruit or other.type <= fruit.type:
            continue
        if abs(other.y - fruit.y) > (other.radius + fruit.radius) * 1.5:
            continue
        reach = other.radius + fruit.radius + MERGE_SLACK
        dx = other.x - fruit.x
        if -reach * 1.25 <= dx < 0:
            if left_big is None or other.x > left_big.x:
                left_big = other
        elif 0 < dx <= reach * 1.25:
            if right_big is None or other.x < right_big.x:
                right_big = other
    return left_big is not None and right_big is not None


def _clears_wedged(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    held_r: float,
    merges: int,
) -> bool:
    """落下が、挟まった同種の合成になっているか。"""
    if merges < 1:
        return False
    for fruit in fruits:
        if fruit.type != drop_type or not _is_wedged(fruit, fruits):
            continue
        if abs(x - fruit.x) <= fruit.radius + held_r + MERGE_SLACK:
            return True
    return False


def _wedged_priority(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    cleared_wedge: bool,
) -> float:
    """大きい実に挟まった同種は、並びより先に大きくする。"""
    has_wedge = any(f.type == drop_type and _is_wedged(f, fruits) for f in fruits)
    if not has_wedge:
        return 0.0
    if cleared_wedge:
        return 220.0
    return -220.0


def _board_score(fruits: list[Fruit], merges: int, *, land_y: float) -> float:
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
    score -= _size_order_penalty(fruits)
    # 危険な山があるときは平坦化より低所へ。横付けで高さを「揃え」に行かない。
    variance = _height_variance(fruits)
    if crown < DANGER_Y:
        variance *= 0.15
    score -= 1.2 * variance
    return score


def _larger_neighbor_bonus(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    held_r: float,
    land_y: float,
    *,
    grow_target: int | None = None,
) -> float:
    """一段大きい実との関係。空いた「並ぶ側」＞上＞逆側。

    大きい順は左＝大・右＝小。オレンジはリンゴの右側の床を最優先し、
    右が塞がっているときだけ上に積む。

    ただし held と next が同種で一段大きい実を育てる局面では、その実の
    「上」を並ぶ側より優先する (二個目を載せて合成→育成)。
    """
    supports = [f for f in fruits if 1 <= f.type - drop_type <= 2]
    if not supports:
        return 0.0

    best = 0.0
    for support in supports:
        gap = support.type - drop_type
        side_x = _ordered_side_x(support, drop_type, held_r)
        side_free = _side_slot_free(fruits, support, side_x, held_r)
        on_top = _is_on_top(support, x, held_r, land_y)
        beside = abs(x - side_x) <= max(held_r, MERGE_SLACK)
        growing = grow_target is not None and support.type == grow_target

        if growing and on_top:
            # 同種 next で育成する対象の上。並ぶ側＋低所着地より強くする。
            best = max(best, 330.0 if gap == 1 else 150.0)
            continue

        if beside and side_free:
            best = max(best, 200.0 if gap == 1 else 90.0)
            continue

        if on_top:
            if side_free:
                # 隣が空いているのに上は弱い。
                best = max(best, 35.0 if gap == 1 else 15.0)
            else:
                best = max(best, 150.0 if gap == 1 else 70.0)
            continue

        if abs(x - support.x) <= support.radius + held_r + MERGE_SLACK:
            best = max(best, 25.0 if gap == 1 else 10.0)

    return best


def _ignored_larger_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    held_r: float,
    land_y: float,
) -> float:
    """一段大きい実があるのに、並ぶ側にも上にも置かないときの減点。"""
    supports = [f for f in fruits if f.type - drop_type == 1]
    if not supports:
        return 0.0

    for support in supports:
        side_x = _ordered_side_x(support, drop_type, held_r)
        if abs(x - side_x) <= max(held_r, MERGE_SLACK):
            return 0.0
        if _is_on_top(support, x, held_r, land_y):
            return 0.0
    return 110.0


def _ordered_side_x(support: Fruit, drop_type: int, held_r: float) -> float:
    """大きい順で隣に並ぶ列。小さい実は大きい実の右。"""
    if drop_type < support.type:
        return support.x + support.radius + held_r
    return support.x - support.radius - held_r


def _side_slot_free(
    fruits: list[Fruit] | tuple[Fruit, ...],
    support: Fruit,
    side_x: float,
    held_r: float,
) -> bool:
    """並ぶ側の床が空いているか (支え以外に邪魔が無い)。"""
    if side_x < held_r or side_x > NORMALIZED_WIDTH - held_r:
        return False
    land = _land_y_excluding(fruits, side_x, held_r, exclude=support)
    floor = NORMALIZED_HEIGHT - held_r
    return land >= floor - 4.0


def _land_y_excluding(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
    *,
    exclude: Fruit,
) -> float:
    return _land_y((f for f in fruits if f is not exclude), x, held_r)


def _is_on_top(support: Fruit, x: float, held_r: float, land_y: float) -> bool:
    """support のほぼ真上に着地しているか。"""
    if abs(x - support.x) > support.radius * 0.85:
        return False
    top = support.y - support.radius
    return abs((land_y + held_r) - top) <= MERGE_SLACK


def _ideal_x(fruit_type: int) -> float:
    """大きいほど左。type 0 が右端寄り。"""
    return NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))


def _anchor_x(drop_type: int, fruits: list[Fruit] | tuple[Fruit, ...], held_r: float) -> float:
    """置きたい列。一段大きい実の並ぶ側が空ならそこ、塞がりなら真上。

    大きい支えが無いときは ideal を、右の小さい実との中間列を残す位置へ寄せる。
    """
    supports = [f for f in fruits if 1 <= f.type - drop_type <= 2]
    if supports:
        support = min(supports, key=lambda f: f.x)
        side_x = _ordered_side_x(support, drop_type, held_r)
        if _side_slot_free(fruits, support, side_x, held_r):
            return side_x
        return support.x

    x = _ideal_x(drop_type)
    for other in fruits:
        if other.type >= drop_type:
            continue
        need = _chain_center_gap(drop_type, other.type)
        x = min(x, other.x - need)
    return max(held_r, min(x, NORMALIZED_WIDTH - held_r))


def _size_order_penalty(fruits: list[Fruit]) -> float:
    """左右の大小が逆転しているペアを減点。絶対 ideal より相対順を見る。"""
    if not fruits:
        return 0.0
    penalty = 0.0
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            if left.type < right.type:
                penalty += (right.type - left.type) * 12.0
    penalty += sum(abs(f.x - _ideal_x(f.type)) for f in fruits) / len(fruits) * 0.12
    return penalty


def _after_drop(obs: Observation, x: float) -> tuple[list[Fruit], int]:
    """テスト用。held を列 x に落としたあとの盤面と合成回数。"""
    assert obs.held_type is not None
    return _simulate_drop(obs.fruits, obs.held_type, x)


def _simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int]:
    placed = list(fruits)
    placed, dropped = _place(placed, fruit_type, x)
    return _resolve_merges(placed, active={dropped})


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
    """合成候補を異種で埋める度合い。小さい実を大きい実の上に載せるのは減点しない。"""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type == under.type:
                continue
            if over.type < under.type:
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
    """列 x に落としたときの中心 y。床か、円どうしが接する位置。

    横ずれがあると大きい実の側面を滑るので、隙間に落ちた小さい実へ届く。
    """
    best = float(NORMALIZED_HEIGHT) - held_r
    for fruit in fruits:
        dx = abs(fruit.x - x)
        gap = fruit.radius + held_r
        if dx >= gap:
            continue
        dy = math.sqrt(gap * gap - dx * dx)
        best = min(best, fruit.y - dy)
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
