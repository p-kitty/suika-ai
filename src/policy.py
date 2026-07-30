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
# 着地後の転がり。側面に乗ったら谷まで横へずらす。
SETTLE_STEP = 3.0
SETTLE_MAX_ITERS = 48
# 隣に並べるとき、ぴったり接触だと肩に乗って弾かれるので少し隙間を空ける。
SIDE_CLEARANCE = 4.0
# 同種ペアの外側に当てて押し込み合成できそうな着地の加点。
PUSH_MERGE_BONUS = 160.0
# 押し込み理想列への近さ (この距離以内で加点)。
PUSH_ALIGN_RANGE = 36.0
# 狙い誤差で内側に入ると外すので、接触より少し外側を狙う。
PUSH_OUTSET = 14.0
# 異種の中央真上は崩壊しやすいので、大側へこの分だけ寄せた列を見る。
LARGE_SIDE_BIAS = 0.4
# 異種のほぼ中央真上への減点。
FOREIGN_CENTER_PENALTY = 140.0
# 大小逆転ペアの type 差あたり減点。
SIZE_ORDER_PAIR_WEIGHT = 28.0
# 各実の ideal 列からの平均距離あたり減点。
SIZE_ORDER_IDEAL_WEIGHT = 0.35
# 崩れた大小順を、中〜大 held で大側端へ押し戻す加点 (push merge 未満)。
RESTORE_ORDER_BONUS = 110.0
# これ未満の held では掃かない (cherry/strawberry/grape)。
RESTORE_MIN_TYPE = 3
# 自分より2段階以上大きい実どうしの隙間に詰める減点。
GAP_JUNK_PENALTY = 200.0


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
    sign = _order_sign(fruits)
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    xs = {round(x / CANDIDATE_STEP) * CANDIDATE_STEP for x in _frange(lo, hi, CANDIDATE_STEP)}
    xs.add(_ideal_x(drop_type, sign))

    for fruit in fruits:
        # 同種、または少し大きい実の上／横 (オレンジ→リンゴなど)。
        if fruit.type < drop_type or fruit.type > drop_type + 2:
            continue
        xs.add(fruit.x)
        # 異種は中央真上より大側寄せの列を候補に入れる。
        if fruit.type != drop_type:
            xs.add(_large_side_x(fruit, sign, held_r))
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

    # 小側の小さい実／大側の大きい実との間に、中間段階の列を残す位置。
    for fruit in fruits:
        if fruit.type < drop_type:
            xs.add(fruit.x - sign * _chain_center_gap(drop_type, fruit.type))
            xs.add(fruit.x - sign * (_chain_center_gap(drop_type, fruit.type) + SIDE_CLEARANCE))
        elif fruit.type > drop_type:
            xs.add(fruit.x + sign * _chain_center_gap(fruit.type, drop_type))
            xs.add(fruit.x + sign * (_chain_center_gap(fruit.type, drop_type) + SIDE_CLEARANCE))

    beside = _smaller_neighbor_x(fruits, drop_type, held_r, sign)
    if beside is not None:
        xs.add(beside)

    # 同種ペアを、held と別種で外側から押す列。
    for _outer, push_x in _push_pair_outers(fruits, drop_type, held_r):
        xs.add(push_x)

    # 大小逆転している実の小側外側 (大側端へ押し戻す列)。
    for _victim, push_x in _restore_push_targets(fruits, drop_type, held_r, sign):
        xs.add(push_x)

    return [x for x in xs if lo <= x <= hi]


def _score(obs: Observation, x: float, held_r: float) -> float:
    """held を落とした盤＋ next の仮想最善手を採点する。"""
    assert obs.held_type is not None
    before = list(obs.fruits)
    sign = _order_sign(before)
    land_x, land_y = _preview_land(before, obs.held_type, x, held_r)
    after, merges = _simulate_drop(before, obs.held_type, x)
    cleared_wedge = _clears_wedged(before, land_x, obs.held_type, held_r, merges)
    grow_target = _growth_target_type(obs.held_type, obs.next_type)

    score = _board_score(after, merges, land_y=land_y, sign=sign)
    score += _wedged_priority(before, obs.held_type, cleared_wedge)
    if merges == 0:
        score += _larger_neighbor_bonus(
            before,
            land_x,
            obs.held_type,
            held_r,
            land_y,
            drop_x=x,
            grow_target=grow_target,
            sign=sign,
        )
        # 挟まった同種を合成する手では、大きい実への寄りを強制しない。
        if not cleared_wedge:
            score -= _ignored_larger_penalty(
                before, land_x, obs.held_type, held_r, land_y, drop_x=x, sign=sign
            )
    else:
        # 合成は同種側。異種の大側寄せで列を盗ませない。大側着地だけ見る。
        score += _merge_large_side_bonus(before, land_x, obs.held_type, sign)
    # 同種以外の中央真上は崩壊しやすいので減点する。
    score -= _foreign_center_penalty(before, x, land_x, land_y, obs.held_type, held_r)

    # 肩に当てて転がした結果、小さい実が大側へ落ちる手を強く落とす。
    # (例: ブドウ左上 → 左へ転がり → 大小順が崩れる)
    # 弾かれて落下列から大きく離れる手も、合成できても減点する。
    score -= _wrong_side_roll_penalty(
        before, land_x, land_y, obs.held_type, held_r, sign
    ) if merges == 0 else 0.0
    score -= _coast_away_penalty(before, x, land_x, land_y, held_r)
    # 大きい実の間に小さいゴミを詰める手。合成では見ない。
    if merges == 0:
        score -= _gap_junk_penalty(before, land_x, land_y, obs.held_type, held_r)

    # 合成が無いときは、一段大きい実の「並ぶ側」寄り。
    # 床に並べるときだけ、中間段階の列潰しを減点する (積み重ねの x は対象外)。
    # 育成で対象の大側に寄せる手は、並ぶ側への引力をかけない。
    # 押し込み合成も、同種の真上 (anchor) に引っ張られない。
    if merges == 0:
        on_grow = grow_target is not None and any(
            f.type == grow_target and _near_support(f, x, land_x, held_r, land_y)
            for f in before
        )
        push = _push_merge_bonus(before, land_x, land_y, obs.held_type, held_r)
        restore = _restore_order_bonus(
            before, land_x, land_y, obs.held_type, held_r, sign
        )
        if not on_grow and push <= 0 and restore <= 0:
            score -= abs(x - _anchor_x(obs.held_type, before, held_r, sign)) * 0.45
        floor = NORMALIZED_HEIGHT - held_r
        if land_y >= floor - 4.0 and not on_grow and push <= 0 and restore <= 0:
            score -= _chain_spacing_penalty(before, land_x, obs.held_type, sign)
        if not _column_fruits(before, x, held_r):
            score += 3.0
        score += push
        score += restore
        if push > 0:
            # 外側の接触列に近い落としを優先 (着地が同じでも狙いを外側へ)。
            score += _push_outer_align(before, x, obs.held_type, held_r)

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
    sign = _order_sign(fruits)
    best = -math.inf
    for nx in _candidates(fruits, next_type, next_r):
        nx = clamp_drop_x(nx, next_type)
        land_x, land_y = _preview_land(fruits, next_type, nx, next_r)
        after, merges = _simulate_drop(fruits, next_type, nx)
        cleared_wedge = _clears_wedged(fruits, land_x, next_type, next_r, merges)
        value = _board_score(after, merges, land_y=land_y, sign=sign)
        value += _wedged_priority(fruits, next_type, cleared_wedge)
        if merges == 0:
            value += _larger_neighbor_bonus(
                fruits, land_x, next_type, next_r, land_y, drop_x=nx, sign=sign
            )
            if not cleared_wedge:
                value -= _ignored_larger_penalty(
                    fruits, land_x, next_type, next_r, land_y, drop_x=nx, sign=sign
                )
        else:
            value += _merge_large_side_bonus(fruits, land_x, next_type, sign)
        value -= _foreign_center_penalty(fruits, nx, land_x, land_y, next_type, next_r)
        if merges == 0:
            value -= _wrong_side_roll_penalty(
                fruits, land_x, land_y, next_type, next_r, sign
            )
            value -= _gap_junk_penalty(fruits, land_x, land_y, next_type, next_r)
            on_grow = any(
                f.type == next_type + 1
                and _near_support(f, nx, land_x, next_r, land_y)
                for f in fruits
            ) and next_type + 1 <= MAX_FRUIT_TYPE
            restore = _restore_order_bonus(
                fruits, land_x, land_y, next_type, next_r, sign
            )
            # next 単体では held/next 同種の育成フラグが無いので、一段大きい実のそばだけ免除。
            if not on_grow and restore <= 0:
                value -= abs(nx - _anchor_x(next_type, fruits, next_r, sign)) * 0.45
                floor = NORMALIZED_HEIGHT - next_r
                if land_y >= floor - 4.0:
                    value -= _chain_spacing_penalty(fruits, land_x, next_type, sign)
            if not _column_fruits(fruits, nx, next_r):
                value += 3.0
            value += restore
        value -= _coast_away_penalty(fruits, nx, land_x, land_y, next_r)
        if value > best:
            best = value
    return 0.0 if best == -math.inf else best


def _chain_center_gap(left_type: int, right_type: int) -> float:
    """大きい側と小さい側の間に、中間段階を全部並べるときの中心距離。"""
    gap = _radius(left_type) + _radius(right_type)
    for mid in range(right_type + 1, left_type):
        gap += 2.0 * _radius(mid)
    return gap


def _chain_spacing_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    sign: int = 1,
) -> float:
    """大きい順の列で、中間段階の隙間を潰す置きを減点する。

    sign=+1 なら左＝大・右＝小。sign=-1 ならその逆。
    """
    penalty = 0.0
    for other in fruits:
        if other.type < drop_type:
            # 小さい実は小側にあるべき。
            on_small_side = (other.x - x) * sign > 0
            if not on_small_side:
                continue
            need = _chain_center_gap(drop_type, other.type)
            lo_x, hi_x = (x, other.x) if x < other.x else (other.x, x)
            for mid in range(other.type + 1, drop_type):
                if any(lo_x < f.x < hi_x and f.type == mid for f in fruits):
                    need -= 2.0 * _radius(mid)
            have = abs(other.x - x)
            if have < need:
                penalty += (need - have) * CHAIN_SPACING_WEIGHT
        elif other.type > drop_type:
            on_large_side = (other.x - x) * sign < 0
            if not on_large_side:
                continue
            need = _chain_center_gap(other.type, drop_type)
            lo_x, hi_x = (x, other.x) if x < other.x else (other.x, x)
            for mid in range(drop_type + 1, other.type):
                if any(lo_x < f.x < hi_x and f.type == mid for f in fruits):
                    need -= 2.0 * _radius(mid)
            have = abs(other.x - x)
            if have < need:
                penalty += (need - have) * CHAIN_SPACING_WEIGHT
    return penalty


def _is_wedged(fruit: Fruit, fruits: list[Fruit] | tuple[Fruit, ...]) -> bool:
    """左右に自分より大きい実が近接して挟まっている。

    大きい実の上に載っているだけの状態は挟まりではない。
    """
    for other in fruits:
        if other is fruit or other.type <= fruit.type:
            continue
        if _is_on_top(other, fruit.x, fruit.radius, fruit.y):
            return False

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
    drop_x: float | None = None,
    grow_target: int | None = None,
    sign: int = 1,
) -> float:
    """一段大きい実との関係。空いた「並ぶ側」＞大側寄せ＞中央真上。

    大きい順の向き (sign) に合わせて隣を選ぶ。sign=+1 なら小は大の右。

    異種の中央真上は加点しない (同種合成以外にメリットが無く崩壊しやすい)。
    held/next 同種の育成では、対象の大側寄せを並ぶ側より優先する。
    """
    supports = [f for f in fruits if 1 <= f.type - drop_type <= 2]
    if not supports:
        return 0.0

    aim_x = x if drop_x is None else drop_x
    best = 0.0
    for support in supports:
        gap = support.type - drop_type
        side_x = _ordered_side_x(support, drop_type, held_r, sign)
        other_x = _ordered_side_x(support, drop_type, held_r, -sign)
        side_free = _side_slot_free(fruits, support, side_x, held_r)
        other_free = _side_slot_free(fruits, support, other_x, held_r)
        on_top = _is_on_top(support, x, held_r, land_y)
        beside = abs(aim_x - side_x) <= max(held_r, MERGE_SLACK)
        beside_other = abs(aim_x - other_x) <= max(held_r, MERGE_SLACK)
        toward_large = _toward_large(support, aim_x, sign)
        near = _near_support(support, aim_x, x, held_r, land_y)
        growing = grow_target is not None and support.type == grow_target
        centered = abs(aim_x - support.x) <= support.radius * 0.2

        if growing and beside and side_free:
            # 育成は空きの並ぶ側が最優先 (大小順を崩さない)。
            best = max(best, 330.0 if gap == 1 else 150.0)
            continue

        if beside and side_free:
            best = max(best, 200.0 if gap == 1 else 90.0)
            continue

        if beside_other and other_free:
            # 並ぶ側が塞がっているときの大側床。
            best = max(best, 180.0 if gap == 1 else 80.0)
            continue

        if near and toward_large and not centered:
            # 大側肩／大側寄り。中央真上より良い。並ぶ側が空なら上の枝で勝つ。
            if growing:
                best = max(best, 300.0 if gap == 1 else 140.0)
            else:
                best = max(best, 170.0 if gap == 1 else 75.0)
            continue

        if on_top and centered:
            # 異種の中央真上は加点しない。
            continue

        if near:
            best = max(best, 25.0 if gap == 1 else 10.0)

    return best


def _ignored_larger_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    held_r: float,
    land_y: float,
    *,
    drop_x: float | None = None,
    sign: int = 1,
) -> float:
    """一段大きい実があるのに、並ぶ側にも大側にも置かないときの減点。

    異種の中央真上は「対処した」とみなさない。
    """
    supports = [f for f in fruits if f.type - drop_type == 1]
    if not supports:
        return 0.0

    aim_x = x if drop_x is None else drop_x
    for support in supports:
        side_x = _ordered_side_x(support, drop_type, held_r, sign)
        other_x = _ordered_side_x(support, drop_type, held_r, -sign)
        if abs(aim_x - side_x) <= max(held_r, MERGE_SLACK):
            return 0.0
        if abs(aim_x - other_x) <= max(held_r, MERGE_SLACK):
            return 0.0
        if (
            _near_support(support, aim_x, x, held_r, land_y)
            and _toward_large(support, aim_x, sign)
            and abs(aim_x - support.x) > support.radius * 0.2
        ):
            return 0.0
    return 110.0


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


def _merge_large_side_bonus(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    drop_type: int,
    sign: int,
) -> float:
    """同種合成の着地が、相手より大側なら加点・小側なら減点。"""
    mates = [f for f in fruits if f.type == drop_type]
    if not mates:
        return 0.0
    mate = min(mates, key=lambda f: abs(f.x - land_x))
    return (land_x - mate.x) * (-sign) * 3.0


def _large_side_x(support: Fruit, sign: int, held_r: float) -> float:
    """支えの大側へ寄せた列。sign=+1 なら大は左なので負方向。"""
    return support.x - sign * min(held_r, support.radius * LARGE_SIDE_BIAS)


def _toward_large(support: Fruit, x: float, sign: int) -> bool:
    """x が support より大側か。"""
    return (x - support.x) * (-sign) > 0


def _near_support(
    support: Fruit,
    drop_x: float,
    land_x: float,
    held_r: float,
    land_y: float,
) -> bool:
    """落下列または着地が支えの近くか。"""
    reach = support.radius + held_r + MERGE_SLACK
    if abs(drop_x - support.x) <= reach or abs(land_x - support.x) <= reach:
        return True
    return _is_on_top(support, land_x, held_r, land_y)


def _wrong_side_roll_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
    sign: int,
) -> float:
    """転がって大きい実の大側床に落ちたときの減点。

    ブドウの左肩に置く → 左へ転がる → イチゴがブドウの左、という崩しを防ぐ。
    """
    floor = NORMALIZED_HEIGHT - held_r
    if land_y < floor - 4.0:
        return 0.0

    penalty = 0.0
    for other in fruits:
        if other.type <= drop_type:
            continue
        # sign=+1 なら小は大より右。land が大より左 (大側) なら崩れ。
        if (land_x - other.x) * sign >= 0:
            continue
        # 肩から転がってすぐ横に落ちたときだけ。遠くの無関係な大実は見ない。
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
    """接触で弾かれて落下列から大きく離れた着地を減点する。

    イチゴ左隣のつもりが左端まで滑る、のような手。
    """
    floor = NORMALIZED_HEIGHT - held_r
    drifted = abs(land_x - drop_x)
    if drifted < held_r * 2:
        return 0.0
    # 床まで滑った距離が大きいほど減点。
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
    """自分より2段階以上大きい実どうしの隙間に小さい実を詰める減点。

    床のくぼみを平坦化したくてチェリーをナシとリンゴの間へ入れる、のような手。
    一段差の並ぶ側 (オレンジ↔リンゴ) は対象外。
    """
    floor = NORMALIZED_HEIGHT - held_r
    if land_y < floor - 4.0:
        return 0.0

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

    # 隣がどちらも held より2段階以上大きいときだけ「ゴミ詰め」。
    if min(left_big.type, right_big.type) - drop_type < 2:
        return 0.0

    sep = right_big.x - left_big.x
    touch = left_big.radius + right_big.radius
    # すでに密着、または広すぎて「間」ではない床は除外。
    if sep <= touch or sep > touch + held_r * 2.8 + MERGE_SLACK:
        return 0.0
    return GAP_JUNK_PENALTY


def _push_pair_outers(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    held_r: float,
) -> list[tuple[Fruit, float]]:
    """押し込み対象の (外側の実, 落としたい列)。held と同種ペアは除外。"""
    outers: list[tuple[Fruit, float]] = []
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if a.type == drop_type or a.type != b.type or _touching(a, b):
                continue
            sep = abs(a.x - b.x)
            need = a.radius + b.radius
            if sep <= need or sep > need + held_r * 2.2:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            outers.append(
                (left, max(lo, left.x - (left.radius + held_r) - PUSH_OUTSET))
            )
            outers.append(
                (right, min(hi, right.x + (right.radius + held_r) + PUSH_OUTSET))
            )
    return outers


def _push_merge_bonus(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
) -> float:
    """別種 held で同種ペアの外側に当て、押し込み合成できそうなら加点する。

    シミュレーションは他実を動かさないので、接触方向と間隔だけで見る。
    """
    best = 0.0
    for outer, ideal_x in _push_pair_outers(fruits, drop_type, held_r):
        # 真上は押し込みではない。
        if _is_on_top(outer, land_x, held_r, land_y):
            continue
        # 着地が外側実の、ペアと反対側に接している。
        if ideal_x < outer.x:
            # 左外側から押す。
            if land_x > outer.x - outer.radius * 0.45:
                continue
        else:
            # 右外側から押す。
            if land_x < outer.x + outer.radius * 0.45:
                continue
        if abs(land_x - outer.x) > outer.radius + held_r + MERGE_SLACK:
            continue
        if land_y + held_r < outer.y - outer.radius - MERGE_SLACK:
            continue
        best = max(best, PUSH_MERGE_BONUS)
    return best


def _push_outer_align(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_x: float,
    drop_type: int,
    held_r: float,
) -> float:
    """押し込みの理想列 (外側接触) への近さ。外れは 0 (減点しない)。"""
    best = 0.0
    for _outer, ideal_x in _push_pair_outers(fruits, drop_type, held_r):
        best = max(best, max(0.0, PUSH_ALIGN_RANGE - abs(drop_x - ideal_x)))
    return best


def _has_size_inversion(
    fruits: list[Fruit] | tuple[Fruit, ...],
    sign: int,
) -> bool:
    """左右の大小が逆転しているペアがあるか。"""
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            if sign > 0 and left.type < right.type:
                return True
            if sign < 0 and left.type > right.type:
                return True
    return False


def _restore_push_targets(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    held_r: float,
    sign: int,
) -> list[tuple[Fruit, float]]:
    """大小逆転ペアの小側側を、さらに小側外側から押す列。"""
    if drop_type < RESTORE_MIN_TYPE or not _has_size_inversion(fruits, sign):
        return []
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    targets: list[tuple[Fruit, float]] = []
    seen: set[int] = set()
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            inverted = (sign > 0 and left.type < right.type) or (
                sign < 0 and left.type > right.type
            )
            if not inverted:
                continue
            # 小側にいる実を大側へ押す。sign=+1 なら右の実を右外側から左へ。
            victim = right if sign > 0 else left
            key = id(victim)
            if key in seen:
                continue
            seen.add(key)
            push_x = victim.x + sign * (victim.radius + held_r + PUSH_OUTSET)
            targets.append((victim, max(lo, min(hi, push_x))))
    return targets


def _restore_order_bonus(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    land_y: float,
    drop_type: int,
    held_r: float,
    sign: int,
) -> float:
    """崩れた大小順を、小側外側から大側端へ押し戻す着地を加点する。

    他実は動かさないシミュレーションなので、接触方向だけで見る。
    異種の中央真上は対象外。push merge より弱く保つ。
    """
    best = 0.0
    for victim, _push_x in _restore_push_targets(fruits, drop_type, held_r, sign):
        if _is_on_top(victim, land_x, held_r, land_y):
            continue
        if sign > 0:
            # 右外側から左へ押す。
            if land_x < victim.x + victim.radius * 0.45:
                continue
        else:
            # 左外側から右へ押す。
            if land_x > victim.x - victim.radius * 0.45:
                continue
        if abs(land_x - victim.x) > victim.radius + held_r + MERGE_SLACK:
            continue
        if land_y + held_r < victim.y - victim.radius - MERGE_SLACK:
            continue
        best = max(best, RESTORE_ORDER_BONUS)
    return best


def _ordered_side_x(
    support: Fruit,
    drop_type: int,
    held_r: float,
    sign: int = 1,
) -> float:
    """大きい順で隣に並ぶ列。sign=+1 なら小さい実は大きい実の右。

    半径和ぴったりだと落下中に肩へ乗って弾かれやすいので、少し隙間を空ける。
    """
    gap = support.radius + held_r + SIDE_CLEARANCE
    if drop_type < support.type:
        return support.x + sign * gap
    return support.x - sign * gap


def _smaller_neighbor_x(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    held_r: float,
    sign: int,
) -> float | None:
    """小側にいる小さい実のすぐ大側に並ぶ列。無ければ None。"""
    smallers = [f for f in fruits if f.type < drop_type]
    if not smallers:
        return None
    # 大側に一番近い小さい実 (sign=+1 なら一番左の小さい実)。
    neighbor = min(smallers, key=lambda f: f.x * sign)
    gap = _chain_center_gap(drop_type, neighbor.type) + SIDE_CLEARANCE
    return neighbor.x - sign * gap


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


def _ideal_x(fruit_type: int, sign: int = 1) -> float:
    """sign=+1 なら大きいほど左。sign=-1 なら大きいほど右。"""
    base = NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))
    if sign < 0:
        return NORMALIZED_WIDTH - base
    return base


def _order_sign(fruits: list[Fruit] | tuple[Fruit, ...]) -> int:
    """盤面の大小の向き。+1=左大右小、-1=左小右大。

    すでに大きい実が右に寄っているのに左を大きくし直さないための判定。
    """
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


def _anchor_x(
    drop_type: int,
    fruits: list[Fruit] | tuple[Fruit, ...],
    held_r: float,
    sign: int = 1,
) -> float:
    """置きたい列。一段大きい実の並ぶ側が空ならそこ、塞がりなら大側寄せ。

    異種の中央真上には錨を置かない。大きい支えが無いときは、小側の
    小さい実のすぐ隣を優先する (ideal まで空けて弾かれを避ける)。
    """
    supports = [f for f in fruits if 1 <= f.type - drop_type <= 2]
    if supports:
        # 大側にある支えを優先 (sign=+1 なら左、sign=-1 なら右)。
        support = min(supports, key=lambda f: f.x * sign)
        side_x = _ordered_side_x(support, drop_type, held_r, sign)
        if _side_slot_free(fruits, support, side_x, held_r):
            return side_x
        other_x = _ordered_side_x(support, drop_type, held_r, -sign)
        if _side_slot_free(fruits, support, other_x, held_r):
            return other_x
        return _large_side_x(support, sign, held_r)

    beside = _smaller_neighbor_x(fruits, drop_type, held_r, sign)
    if beside is not None:
        return max(held_r, min(beside, NORMALIZED_WIDTH - held_r))

    return max(held_r, min(_ideal_x(drop_type, sign), NORMALIZED_WIDTH - held_r))


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
            # sign=+1: 左が大きいべき。sign=-1: 左が小さいべき。
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
    """列 x にフルーツを着地させて追加する。追加した index も返す。

    側面に当たったあとの転がりを入れてから置く。合成で出した実は
    惰性滑りなし (中点付近に落ち着く) にする。
    """
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
            # 支点より右に乗っていればさらに右へ転がり落ちる。
            push += dx

        if abs(push) < 0.75:
            return x

        coast_dir = math.copysign(1.0, push)
        nxt = max(lo, min(hi, x + coast_dir * SETTLE_STEP))
        nxt_y = _land_y(fruits, nxt, held_r)
        # y は下向き。小さくなったら上りなので止める。
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
    # 空いた床を横断して壁まで届く長さ。
    max_iters = int(NORMALIZED_WIDTH / SETTLE_STEP) + 5

    for _ in range(max_iters):
        nxt = max(lo, min(hi, x + direction * SETTLE_STEP))
        if abs(nxt - x) < 1e-6:
            return x
        nxt_y = _land_y(fruits, nxt, held_r)
        # 他実の斜面に乗り上げる直前で止める。
        if nxt_y < floor - 1.0:
            return x
        # 床上で他実にめり込むなら接触位置へ。
        for fruit in fruits:
            limit = fruit.radius + held_r
            if abs(nxt - fruit.x) < limit - 0.5:
                if direction > 0:
                    return max(lo, min(hi, fruit.x - limit))
                return max(lo, min(hi, fruit.x + limit))
        x = nxt
    return x


def _resolve_merges(fruits: list[Fruit], active: set[int]) -> tuple[list[Fruit], int]:
    """落とした実から始まる同種接触だけを合成する。観測盤は静止前提。

    合成実は中点に出し、その後 `_place` 経由で転がして着地する。
    """
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
    """合成候補を異種で埋める度合い。小さい実を大きい実の上に載せるのは減点しない。"""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type == under.type:
                continue
            if over.type < under.type:
                continue
            # y は下向き。埋める側 over は under より上 (小さい y)。
            if over.y >= under.y:
                continue
            if abs(over.x - under.x) > under.radius * 0.9:
                continue
            # under の頭頂と over の下端の隙間。
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
