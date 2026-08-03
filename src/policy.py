"""落とす列を決める。薄い bootstrap 方策 (RL の土台)。

具体手順 (押し込み・復元押し・連鎖隙間空け・梯子発火など) は持たない。
合成・危険高さ・埋め込み・薄い大小順・転がり事故防止だけ見る。
大きい実どうしは近接、大側端の角ポケット (L 中心より下) は避ける。
谷育成は同種待ちか held/next が両方とも壁よりひとつ小さいときに限る。
手の採点は eval = score (本家の合成点) - penalties (事故・悪手の減点)。
"""

from __future__ import annotations

import math
import statistics

from .observe import Observation, clamp_drop_x
from .reward import merge_score
from .sim_physics import landed_xy
from .sim_physics import simulate_drop
from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE, SPAWN_MAX_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# --- 複数箇所で共有するチューニング ---
# 合成できそうな接触の許容 (中心距離と半径和の差)。
MERGE_SLACK = 18.0
# next 手の割引。
NEXT_DISCOUNT = 0.55
# 異種真上とみなす着地の横ずれ (下実半径に対する割合)。
FOREIGN_AIM_CENTER_FRAC = 0.20
# 真下の異種中心帯に着地したときの減点。
FOREIGN_AIM_PENALTY = 100.0
# 探索の粗さ。物理 (simulate_drop) が支配的で、ここが実行時間をほぼ決める。
# 旧 8/16 は 1 手 3.8 秒で収集が回らない。1.2 秒 / score -3.4% で取り引き。
# next 先読みを回す held 候補の本数。物理が重いので上位だけ。
HELD_TOP = 2
# next 先読みの候補刻み。held (CANDIDATE_STEP) より粗い。
NEXT_CANDIDATE_STEP = 32.0
# held 候補の均等刻み。粗くすると危険な山の真上が候補に乗るので下げない
# (20 で test_avoids_dangerous_tall_stack が落ちた)。速度は先読み側で稼ぐ。
CANDIDATE_STEP = 12.0


def choose_x(obs: Observation) -> float:
    """観測から落とす列を返す。ready で held_type がある前提。"""
    if obs.held_type is None:
        raise ValueError("held_type が無い")

    held_r = fruit_radius(obs.held_type)
    ranked: list[tuple[float, float, list[Fruit]]] = []
    for x in _candidates(list(obs.fruits), obs.held_type, held_r, extra_type=obs.next_type):
        x = clamp_drop_x(x, obs.held_type)
        after, held_eval = _held_eval(obs, x, held_r)
        ranked.append((held_eval, x, after))

    if not ranked:
        return NORMALIZED_WIDTH / 2

    ranked.sort(key=lambda row: row[0], reverse=True)
    if obs.next_type is None:
        return ranked[0][1]

    # next 先読みは held の eval 上位だけ (物理が重い)。候補は held より粗い刻み。
    best_x = ranked[0][1]
    best_score = -math.inf
    for held_eval, x, after in ranked[:HELD_TOP]:
        value = held_eval + NEXT_DISCOUNT * _best_next_score(
            after, obs.next_type, step=NEXT_CANDIDATE_STEP
        )
        if value > best_score:
            best_score = value
            best_x = x
    return best_x


def _candidates(
    fruits: tuple[Fruit, ...] | list[Fruit],
    drop_type: int,
    held_r: float,
    extra_type: int | None = None,
    *,
    step: float | None = None,
) -> list[float]:
    """均等刻みに、同種・近い実の上／横と ideal_x を足す。"""
    sign = _order_sign(fruits)
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    grid = CANDIDATE_STEP if step is None else step
    xs = {round(x / grid) * grid for x in _frange(lo, hi, grid)}
    xs.add(_ideal_x(drop_type, sign))
    _add_near_fruit_x(xs, fruits, held_r, lambda t: drop_type <= t <= drop_type + 2)

    if extra_type is not None:
        xs.add(_ideal_x(extra_type, sign))
        _add_near_fruit_x(xs, fruits, held_r, lambda t: t == extra_type)

    return [x for x in xs if lo <= x <= hi]


def _add_near_fruit_x(
    xs: set[float],
    fruits: tuple[Fruit, ...] | list[Fruit],
    held_r: float,
    matches,
) -> None:
    """type が matches を満たす実の上／左右接触位置を xs に足す。"""
    for fruit in fruits:
        if not matches(fruit.type):
            continue
        xs.add(fruit.x)
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)


def drop_scores(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    *,
    next_type: int | None = None,
) -> tuple[float, float, float, list[Fruit], int]:
    """列 x に落とした 1 手の (score, penalties, eval, after, merges)。

    sim / 学習用。after と merges は simulate_drop の結果をそのまま返す。
    盤面ぶんの減点は落下前との差にする。同じ盤では定数差なので choose_x の
    選び方は変わらず、手ごとに足しても盤の大きさで膨らまない。
    """
    held_r = fruit_radius(drop_type)
    before = list(fruits)
    after, score, penalties, merges = _evaluate_drop(
        before,
        drop_type,
        clamp_drop_x(x, drop_type),
        held_r,
        next_type=next_type,
    )
    penalties -= _board_penalties(before, sign=_order_sign(before))
    return score, penalties, score - penalties, after, merges


def _held_eval(obs: Observation, x: float, held_r: float) -> tuple[list[Fruit], float]:
    """held を x に落としたあとの (盤面, score - penalties)。next は見ない。"""
    assert obs.held_type is not None
    before = list(obs.fruits)
    after, score, penalties, _merges = _evaluate_drop(
        before, obs.held_type, x, held_r, next_type=obs.next_type
    )
    return after, score - penalties


def _score(obs: Observation, x: float, held_r: float) -> float:
    """held を落とした盤＋ next の仮想最善手を採点する。"""
    after, value = _held_eval(obs, x, held_r)
    if obs.next_type is not None:
        value += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)
    return value


def _best_next_score(
    fruits: list[Fruit],
    next_type: int,
    *,
    step: float | None = None,
) -> float:
    """next を最善列に落としたときの eval。"""
    next_r = fruit_radius(next_type)
    best = -math.inf
    for nx in _candidates(fruits, next_type, next_r, step=step):
        nx = clamp_drop_x(nx, next_type)
        # その先の next は未知。育成免除は谷内同種だけが効く。
        _, score, penalties, _merges = _evaluate_drop(fruits, next_type, nx, next_r)
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
) -> tuple[list[Fruit], float, float, int]:
    """1 手落としたあとの盤面・本家点・減点・合成回数。"""
    before = list(fruits)
    sign = _order_sign(before)
    after, merges, merge_types = simulate_drop(before, drop_type, x)
    land_x, land_y = landed_xy(before, after, drop_type, x, held_r, merges)

    score = merge_score(merge_types)
    penalties = _board_penalties(after, sign=sign)
    # 条件を満たす谷着地だけ育成枠。wrong_side で潰さない。
    growing = _valley_grow_ok(before, land_x, drop_type, next_type)
    # FOREIGN_AIM は merges ではなく「真下の実が異種か」で見る。
    # 同種が真下なら合体待ちで OK。異種真上から転がって床で合体しても減点。
    penalties += _foreign_aim_penalty(before, land_x, land_y, drop_type, held_r)
    if merges == 0:
        if not growing:
            penalties += _wrong_side_roll_penalty(
                before, land_x, land_y, drop_type, held_r, sign
            )
        penalties += _bury_block_penalty(before, land_x, land_y, drop_type, held_r)
    penalties += _coast_away_penalty(before, x, land_x, land_y, held_r)
    return after, score, penalties, merges


def _board_penalties(fruits: list[Fruit], *, sign: int = 1) -> float:
    """落としたあとの盤面減点（危険・埋め込み・同種過多・サイズ順・大寄せ・凸凹）。"""
    danger_y = 90.0
    danger_crown_weight = 0.5
    bury_weight = 20.0
    variance_weight = 0.08
    variance_danger_scale = 0.15

    penalty = 0.0
    crown = _top_crown(fruits)
    if crown < danger_y:
        penalty += (danger_y - crown) * danger_crown_weight

    penalty += bury_weight * _bury_penalty(fruits)
    penalty += _excess_same_penalty(fruits)
    penalty += _size_order_penalty(fruits, sign)
    penalty += _big_layout_penalty(fruits, sign)
    variance = _height_variance(fruits)
    if crown < danger_y:
        variance *= variance_danger_scale
    penalty += variance_weight * variance
    return penalty


def _big_layout_penalty(fruits: list[Fruit] | tuple[Fruit, ...], sign: int = 1) -> float:
    """大実どうしの近接と、大側端の角ポケット減点。

    sign=+1 なら左が大側、-1 なら右が大側。角ポケットはその側だけ見る。
    最大実 L が大側壁に付いているとき、L より外側かつ L.y より下の小実を強く減点。
    """
    if not fruits:
        return 0.0
    max_t = max(fruit.type for fruit in fruits)

    cluster_weight = 0.025
    under_l_weight = 50.0
    big_min = max(0, max_t - 2)
    large_left = sign > 0

    penalty = 0.0
    max_fruits = [fruit for fruit in fruits if fruit.type == max_t]

    for big in max_fruits:
        if large_left:
            wall_gap = big.x - big.radius
        else:
            wall_gap = NORMALIZED_WIDTH - big.radius - big.x
        edge_anchored = max(24.0, big.radius * 0.35)
        if wall_gap > edge_anchored:
            continue
        for fruit in fruits:
            if fruit.type >= max_t:
                continue
            if fruit.y <= big.y:
                continue
            on_outer = fruit.x < big.x if large_left else fruit.x > big.x
            if not on_outer:
                continue
            depth = fruit.y - big.y
            penalty += under_l_weight * (1.0 + 0.05 * (max_t - fruit.type))
            penalty += 0.15 * depth

    bigs = sorted(
        (fruit for fruit in fruits if fruit.type >= big_min),
        key=lambda fruit: fruit.x,
    )
    for i in range(len(bigs) - 1):
        left, right = bigs[i], bigs[i + 1]
        gap = (right.x - left.x) - left.radius - right.radius
        if gap <= 0:
            continue
        gap = min(gap, left.radius + right.radius)
        size = 0.5 + 0.05 * (left.type + right.type)
        penalty += cluster_weight * gap * size
    return penalty


def _excess_same_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """同種が 3 個以上ある超過分を減点。2 個までは合成待ちとして許容。"""
    excess_same_weight = 20.0
    counts: dict[int, int] = {}
    for fruit in fruits:
        counts[fruit.type] = counts.get(fruit.type, 0) + 1
    penalty = 0.0
    for count in counts.values():
        if count >= 3:
            penalty += (count - 2) * excess_same_weight
    return penalty


def _fruit_below(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    held_r: float,
) -> Fruit | None:
    """着地の真下で支えている実。床だけなら None。"""
    best: Fruit | None = None
    best_err = math.inf
    for fruit in fruits:
        # 支点は着地より下 (y 大)。
        if fruit.y <= land_y - 1.0:
            continue
        dist = math.hypot(land_x - fruit.x, land_y - fruit.y)
        touch = fruit.radius + held_r
        if dist > touch + MERGE_SLACK:
            continue
        err = abs(dist - touch)
        if err < best_err:
            best_err = err
            best = fruit
    return best


def _foreign_aim_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
) -> float:
    """真下の異種のガチ真上に着地する減点。

    真下が同種なら合体待ちで 0。肩着地や床着地も 0。
    merges は見ない (異種真上から転がって床で合体しても減点する)。
    """
    under = _fruit_below(fruits, land_x, land_y, held_r)
    if under is None or under.type == drop_type:
        return 0.0
    # 中心がずれていれば真上ではない。肩着地は対象外。
    if abs(land_x - under.x) > under.radius * FOREIGN_AIM_CENTER_FRAC:
        return 0.0
    return FOREIGN_AIM_PENALTY


def _wrong_side_roll_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
    sign: int,
) -> float:
    """転がって大きい実の大側床に落ちたときの減点。"""
    wrong_side_base = 8.0
    wrong_side_type_weight = 2.0
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
        penalty += wrong_side_base + wrong_side_type_weight * (other.type - drop_type)
    return penalty


def _coast_away_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_x: float,
    land_x: float,
    land_y: float,
    held_r: float,
) -> float:
    """接触で弾かれて落下列から大きく離れた着地を減点する。"""
    coast_drift_weight = 0.08
    coast_floor_bonus = 8.0
    floor = NORMALIZED_HEIGHT - held_r
    drifted = abs(land_x - drop_x)
    if drifted < held_r * 2:
        return 0.0
    penalty = drifted * coast_drift_weight
    if land_y >= floor - 4.0 and drifted > NORMALIZED_WIDTH * 0.25:
        penalty += coast_floor_bonus
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
    size_order_pair_weight = 1.5
    size_order_ideal_weight = 0.004
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
                penalty += (right.type - left.type) * size_order_pair_weight
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * size_order_pair_weight
    if open_fruits:
        penalty += (
            sum(abs(f.x - _ideal_x(f.type, sign)) for f in open_fruits)
            / len(open_fruits)
            * size_order_ideal_weight
        )
    return penalty


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
    bury_block_weight = 14.0
    bury_shoulder_scale = 0.5
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
        scale = 1.0 if dx <= under.radius * 0.5 else bury_shoulder_scale
        penalty += scale * bury_block_weight * (drop_type - under.type)
    return penalty


def _height_variance(fruits: list[Fruit]) -> float:
    """列ビンごとの頭頂のばらつき。空なら 0。"""
    flat_bin = 40.0
    bins: dict[int, float] = {}
    for fruit in fruits:
        key = int(fruit.x // flat_bin)
        top = fruit.y - fruit.radius
        bins[key] = min(bins.get(key, float(NORMALIZED_HEIGHT)), top)
    if len(bins) < 2:
        return 0.0
    return float(statistics.pstdev(list(bins.values())))


def _frange(start: float, stop: float, step: float):
    x = start
    while x <= stop + 1e-6:
        yield x
        x += step
