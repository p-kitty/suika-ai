"""落とす列を決める。薄い bootstrap 方策 (RL の土台)。

具体手順 (押し込み・復元押し・連鎖隙間空けなど) は持たない。
合成・危険高さ・埋め込み・薄い大小順・転がり事故防止だけ見る。
大きい実の谷への育成は、谷に同種があるときか、held と next が両方とも
壁よりひとつ小さいときに限る (それ以外の隙間埋めは通常減点)。
手の採点は eval = score (本家の合成点) - penalties (事故・悪手の減点)。
"""

from __future__ import annotations

import math
import statistics

from .observe import Observation, clamp_drop_x
from .reward import merge_score
from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE, SPAWN_MAX_TYPE
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
# next 手の割引。
NEXT_DISCOUNT = 0.55
# 着地後の転がり。側面に乗ったら谷まで横へずらす。
SETTLE_STEP = 3.0
SETTLE_MAX_ITERS = 48
# 支持円のほぼ頂点 (不安定) とみなす |dx| / radius。
APEX_DX_FRAC = 0.2
# 減点は本家点 (1〜65) と釣り合うスケールにする。加点は本家点だけ。
# 異種のほぼ中央を狙う減点 (転がって床に落ちても)。実機では崩れやすい。
FOREIGN_AIM_PENALTY = 10.0
# 同種が 3 個以上あるときの超過 1 個あたり減点。2 個までは待ち OK。
EXCESS_SAME_WEIGHT = 20.0
# 大小逆転ペアの type 差あたり減点。
SIZE_ORDER_PAIR_WEIGHT = 1.5
# ideal 列からの平均距離あたり減点 (弱め。レイアウト強制にしない)。
SIZE_ORDER_IDEAL_WEIGHT = 0.004
# ideal 列への弱い引力。
IDEAL_PULL = 0.015
# 床から積み上げた高さあたりの減点 (積み上げを止める最低限)。
LAND_HEIGHT_WEIGHT = 0.05
DANGER_CROWN_WEIGHT = 0.5
BURY_WEIGHT = 20.0
# 同種ペア待ちを、より大きい異種で塞ぐ手の減点 (type 差あたり)。
BURY_BLOCK_WEIGHT = 14.0
# 直上でなく肩から塞いだときの割合。
BURY_SHOULDER_SCALE = 0.5
VARIANCE_WEIGHT = 0.08
VARIANCE_DANGER_SCALE = 0.15
WRONG_SIDE_BASE = 8.0
WRONG_SIDE_TYPE_WEIGHT = 2.0
COAST_DRIFT_WEIGHT = 0.08
COAST_FLOOR_BONUS = 8.0


def choose_x(obs: Observation) -> float:
    """観測から落とす列を返す。ready で held_type がある前提。"""
    if obs.held_type is None:
        raise ValueError("held_type が無い")

    held_r = fruit_radius(obs.held_type)
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


def drop_scores(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    *,
    next_type: int | None = None,
) -> tuple[float, float, float]:
    """列 x に落とした 1 手の (score, penalties, eval)。sim / 学習用。

    盤面ぶんの減点は落下前との差にする。同じ盤では定数差なので choose_x の
    選び方は変わらず、手ごとに足しても盤の大きさで膨らまない。
    """
    held_r = fruit_radius(drop_type)
    before = list(fruits)
    _, score, penalties = _evaluate_drop(
        before,
        drop_type,
        clamp_drop_x(x, drop_type),
        held_r,
        next_type=next_type,
    )
    penalties -= _board_penalties(before, sign=_order_sign(before))
    return score, penalties, score - penalties


def _score(obs: Observation, x: float, held_r: float) -> float:
    """held を落とした盤＋ next の仮想最善手を採点する。"""
    assert obs.held_type is not None
    before = list(obs.fruits)
    after, score, penalties = _evaluate_drop(
        before, obs.held_type, x, held_r, next_type=obs.next_type
    )
    value = score - penalties
    if obs.next_type is not None:
        value += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)
    return value


def _best_next_score(fruits: list[Fruit], next_type: int) -> float:
    """next を最善列に落としたときの eval。"""
    next_r = fruit_radius(next_type)
    best = -math.inf
    for nx in _candidates(fruits, next_type, next_r):
        nx = clamp_drop_x(nx, next_type)
        # その先の next は未知。育成免除は谷内同種だけが効く。
        _, score, penalties = _evaluate_drop(fruits, next_type, nx, next_r)
        if score - penalties > best:
            best = score - penalties
    return 0.0 if best == -math.inf else best


def _evaluate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    held_r: float,
    *,
    next_type: int | None = None,
) -> tuple[list[Fruit], float, float]:
    """1 手落としたあとの盤面・本家点・減点。"""
    before = list(fruits)
    sign = _order_sign(before)
    land_x, land_y = _preview_land(before, drop_type, x, held_r)
    after, merges, merge_types = simulate_drop(before, drop_type, x)

    score = merge_score(merge_types)
    penalties = _board_penalties(after, sign=sign)
    # 条件を満たす谷着地だけ育成枠。高さ・wrong_side・ideal で潰さない。
    growing = _valley_grow_ok(before, land_x, drop_type, next_type)
    if merges == 0:
        # 合成した実は残らないので、積み上げ減点は盤に残る手にだけかける。
        if not growing:
            floor = NORMALIZED_HEIGHT - held_r
            penalties += max(0.0, floor - land_y) * LAND_HEIGHT_WEIGHT
            penalties += _wrong_side_roll_penalty(
                before, land_x, land_y, drop_type, held_r, sign
            )
            penalties += abs(x - _ideal_x(drop_type, sign)) * IDEAL_PULL
        penalties += _foreign_aim_penalty(before, x, drop_type)
        penalties += _bury_block_penalty(before, land_x, land_y, drop_type, held_r)
    penalties += _coast_away_penalty(before, x, land_x, land_y, held_r)
    return after, score, penalties


def _board_penalties(fruits: list[Fruit], *, sign: int = 1) -> float:
    """落としたあとの盤面減点（危険・埋め込み・同種過多・サイズ順・凸凹）。"""
    penalty = 0.0
    crown = _top_crown(fruits)
    if crown < DANGER_Y:
        penalty += (DANGER_Y - crown) * DANGER_CROWN_WEIGHT

    penalty += BURY_WEIGHT * _bury_penalty(fruits)
    penalty += _excess_same_penalty(fruits)
    penalty += _size_order_penalty(fruits, sign)
    variance = _height_variance(fruits)
    if crown < DANGER_Y:
        variance *= VARIANCE_DANGER_SCALE
    penalty += VARIANCE_WEIGHT * variance
    return penalty


def _excess_same_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """同種が 3 個以上ある超過分を減点。2 個までは合成待ちとして許容。"""
    counts: dict[int, int] = {}
    for fruit in fruits:
        counts[fruit.type] = counts.get(fruit.type, 0) + 1
    penalty = 0.0
    for count in counts.values():
        if count >= 3:
            penalty += (count - 2) * EXCESS_SAME_WEIGHT
    return penalty


def _foreign_aim_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_x: float,
    drop_type: int,
) -> float:
    """異種のほぼ中央を狙う減点。転がっても実機では不安定。"""
    for fruit in fruits:
        if fruit.type == drop_type:
            continue
        if abs(drop_x - fruit.x) <= fruit.radius * 0.3:
            return FOREIGN_AIM_PENALTY
    return 0.0


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
        penalty += WRONG_SIDE_BASE + WRONG_SIDE_TYPE_WEIGHT * (other.type - drop_type)
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
    penalty = drifted * COAST_DRIFT_WEIGHT
    if land_y >= floor - 4.0 and drifted > NORMALIZED_WIDTH * 0.25:
        penalty += COAST_FLOOR_BONUS
    return penalty


def _valley_flanks(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
) -> tuple[Fruit, Fruit] | None:
    """x が、drop_type より大きい実どうしの狭い谷に入っているときの左右。"""
    left_big: Fruit | None = None
    right_big: Fruit | None = None
    for fruit in fruits:
        if fruit.type <= drop_type:
            continue
        if fruit.x < x:
            if left_big is None or fruit.x > left_big.x:
                left_big = fruit
        elif fruit.x > x:
            if right_big is None or fruit.x < right_big.x:
                right_big = fruit
    if left_big is None or right_big is None:
        return None
    held_r = fruit_radius(drop_type)
    sep = right_big.x - left_big.x
    touch = left_big.radius + right_big.radius
    if sep > touch + held_r * 2.8 + MERGE_SLACK:
        return None
    return left_big, right_big


def _valley_grow_ok(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    drop_type: int,
    next_type: int | None,
) -> bool:
    """谷への育成免除をしてよいか。

    - 谷の間に同種がある (掃除・合体待ち)
    - または held と next が両方とも、左右の壁よりひとつ小さい
    """
    flanks = _valley_flanks(fruits, land_x, drop_type)
    if flanks is None:
        return False
    left, right = flanks
    for fruit in fruits:
        if fruit.type == drop_type and left.x < fruit.x < right.x:
            return True
    wall = min(left.type, right.type)
    return (
        next_type is not None
        and drop_type == next_type
        and drop_type == wall - 1
    )


def _is_nestled(
    fruit: Fruit,
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> bool:
    """より大きい実どうしの谷に収まっているか。"""
    return _valley_flanks(fruits, fruit.x, fruit.type) is not None


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
        if fruit.type >= SPAWN_MAX_TYPE and fruit.x > NORMALIZED_WIDTH * 0.55:
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
    """左右の大小が逆転しているペアを減点。絶対 ideal より相対順を見る。

    谷に育てている小さい実は大小順の対象外 (育成をレイアウト減点で潰さない)。
    """
    if not fruits:
        return 0.0
    penalty = 0.0
    open_fruits = [f for f in fruits if not _is_nestled(f, fruits)]
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            if _is_nestled(a, fruits) or _is_nestled(b, fruits):
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            if sign > 0 and left.type < right.type:
                penalty += (right.type - left.type) * SIZE_ORDER_PAIR_WEIGHT
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * SIZE_ORDER_PAIR_WEIGHT
    if open_fruits:
        penalty += (
            sum(abs(f.x - _ideal_x(f.type, sign)) for f in open_fruits)
            / len(open_fruits)
            * SIZE_ORDER_IDEAL_WEIGHT
        )
    return penalty


def simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int, list[int]]:
    """列 x に落としたあとの盤面・合成回数・合成元 type 列。sim / 学習用。"""
    placed = list(fruits)
    placed, dropped = _place(placed, fruit_type, x)
    return _resolve_merges(placed, active={dropped})


def _preview_land(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
    held_r: float,
) -> tuple[float, float]:
    """落下列 x から、転がり後の着地 (x, y)。"""
    x = _settle_x(fruits, x, held_r, allow_coast=True, drop_type=fruit_type)
    return x, _land_y(fruits, x, held_r)


def _place(
    fruits: list[Fruit],
    fruit_type: int,
    x: float,
    *,
    allow_coast: bool = True,
) -> tuple[list[Fruit], int]:
    """列 x にフルーツを着地させて追加する。追加した index も返す。"""
    r = fruit_radius(fruit_type)
    x = _settle_x(fruits, x, r, allow_coast=allow_coast, drop_type=fruit_type)
    y = _land_y(fruits, x, r)
    fruits.append(Fruit(type=fruit_type, x=x, y=y, radius=r, confidence=100.0))
    return fruits, len(fruits) - 1


def _settle_x(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
    *,
    allow_coast: bool = True,
    drop_type: int | None = None,
) -> float:
    """円の側面に乗ったら谷・床まで転がし、床では惰性で壁／他実まで滑る。

    異種の支持円のほぼ頂点は不安定なので左右どちらかへ転がす。
    同種の真上は合成のためそのまま着地させる。
    """
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

        # 今の列で同種に届くなら、異種の斜面・頂点で転がして合体を逃さない。
        if drop_type is not None and _would_merge_at(fruits, x, y, held_r, drop_type):
            return x

        push = 0.0
        apex_dx = 0.0
        apex_support_x = x
        on_apex = False
        for fruit in fruits:
            dx = x - fruit.x
            gap = fruit.radius + held_r
            if abs(dx) >= gap - 1e-6:
                continue
            dy = math.sqrt(max(0.0, gap * gap - dx * dx))
            if abs((fruit.y - dy) - y) > 2.0:
                continue
            push += dx
            if drop_type is not None and fruit.type == drop_type:
                continue
            if abs(dx) <= max(fruit.radius * APEX_DX_FRAC, 1.0):
                on_apex = True
                apex_dx = dx
                apex_support_x = fruit.x

        if abs(push) < 0.75:
            if not on_apex:
                return x
            # 頂点のつり合いは壊す。微小 dx があればその向き、無ければ決定論で左右。
            if abs(apex_dx) > 1e-9:
                push = math.copysign(1.0, apex_dx)
            else:
                push = _apex_roll_dir(apex_support_x)

        coast_dir = math.copysign(1.0, push)
        nxt = max(lo, min(hi, x + coast_dir * SETTLE_STEP))
        nxt_y = _land_y(fruits, nxt, held_r)
        if nxt_y < y - 0.5:
            return x
        if abs(nxt - x) < 1e-6:
            return x
        x = nxt
    return x


def _would_merge_at(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    y: float,
    held_r: float,
    drop_type: int,
) -> bool:
    """列 (x, y) に着地した同種が、既存の同種と接触するか。"""
    held = Fruit(type=drop_type, x=x, y=y, radius=held_r, confidence=100.0)
    return any(fruit.type == drop_type and _touching(held, fruit) for fruit in fruits)


def _apex_roll_dir(support_x: float) -> float:
    """真上つり合いを崩す向き。同じ support_x なら毎回同じ。"""
    return 1.0 if int(round(support_x / SETTLE_STEP)) % 2 == 0 else -1.0


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


def _resolve_merges(
    fruits: list[Fruit], active: set[int]
) -> tuple[list[Fruit], int, list[int]]:
    """落とした実から始まる同種接触だけを合成する。観測盤は静止前提。"""
    fruits = list(fruits)
    merges = 0
    merge_types: list[int] = []
    for _ in range(64):
        pair = _find_merge_pair(fruits, active)
        if pair is None:
            break
        i, j = pair
        a, b = fruits[i], fruits[j]
        source_type = a.type
        new_type = source_type + 1
        mid_x = (a.x + b.x) / 2
        for idx in sorted((i, j), reverse=True):
            fruits.pop(idx)

        merge_types.append(source_type)
        merges += 1
        if new_type > MAX_FRUIT_TYPE:
            active = set()
            continue

        fruits, new_i = _place(fruits, new_type, mid_x, allow_coast=False)
        active = {new_i}

    return fruits, merges, merge_types


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


def _bury_block_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
) -> float:
    """同種ペア待ちの実を、より大きい異種で直上・肩から塞ぐ減点。"""
    penalty = 0.0
    for under in fruits:
        if under.type >= drop_type:
            continue
        if not any(f.type == under.type and f is not under for f in fruits):
            continue
        dx = abs(land_x - under.x)
        if dx > under.radius + held_r * 0.5:
            continue
        # 頭に乗っているか。横に並んだだけなら塞いでいない。
        over_top = (land_y + held_r) - (under.y - under.radius)
        if over_top > under.radius:
            continue
        scale = 1.0 if dx <= under.radius * 0.5 else BURY_SHOULDER_SCALE
        penalty += scale * BURY_BLOCK_WEIGHT * (drop_type - under.type)
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


def _frange(start: float, stop: float, step: float):
    x = start
    while x <= stop + 1e-6:
        yield x
        x += step
