"""1 手の採点のうち、減点 (penalties) 側。

`policy.py` の探索から呼ばれる。eval = score (本家の合成点) - penalties のうち、
この方策が「事故・悪手」とみなす形を数える側をここに置く。

依存は一方向 (policy -> penalties)。ここから探索側を呼んではいけない。
盤面の大小の向き (`sign`) は呼び元が決めて引数で渡す。

各定数のコメントは、その値に決まった実測の記録。数字を動かすときは
scripts/compare_policy.py の A/B に掛けてから。`_apply_variant` は
モジュール属性を書き換える方式なので、この中の参照は常に
モジュールグローバル経由で読むこと (`from .penalties import X` で束縛すると
書き換えが効かなくなる)。
"""

from __future__ import annotations

import math

from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# --- 複数箇所で共有するチューニング ---
# 合成できそうな接触の許容 (中心距離と半径和の差)。
MERGE_SLACK = 18.0
# 異種真上とみなす着地の横ずれ (下実半径に対する割合)。
FOREIGN_AIM_CENTER_FRAC = 0.20
# 真下の異種中心帯に着地したときの減点。
FOREIGN_AIM_PENALTY = 100.0
# 同点候補の順位を決めるためだけの重み。他の項が全部並ぶ局面は常態で、これが無いと
# 勝つ手が候補 set の列挙順 = float のハッシュ順という実装詳細で決まってしまう。
# 最小の合成点が 1.0 (cherry -> straw) なので、本物の差を覆さないよう最大でも
# その 5 分の 1 に収める (`|x - 中央|` は高々 190 なので 0.001 で 0.19)。
# 手の良し悪しをこれで表そうとしないこと。
CENTER_TIEBREAK_WEIGHT = 0.001

# 谷育成ボーナス (減点から引く形で入れる)。`valley_grow_ok` が成立する着地だけ。
# 実合体より強くしない。8.0 だと grape 合体 (6 点) を蹴って非合体の谷を選んだ。
# 2.0 で育成側に倒れ、3.0 でも合体は取り続ける (実測)。
VALLEY_GROW_BONUS = 3.0

# --- 壁付き判定 (角ポケット減点と梯子の土台で共有) ---
EDGE_ANCHOR_MIN = 24.0
EDGE_ANCHOR_FRAC = 0.35

# 「大きい実の塊」とみなす最大実からの段数。
BIG_CLUSTER_SPAN = 2

# --- 盤面減点の重み ---
# compare_policy の A/B がモジュール属性として差し替えるので、
# board_penalties のローカルではなくここに置く。
BURY_WEIGHT = 20.0
# 大実の肩に載せてよい型差の上限。パイン (8) の肩にオレンジ (4) までは許す。
PERCH_MIN_GAP = 5
# 肩を見る実の範囲 (最大実から何段下まで)。0 なら最大実だけ。
PERCH_BIG_SPAN = 1
PERCH_WEIGHT = 16.0



# --- 盤面を読むだけの補助 -------------------------------------------------


def ideal_x(fruit_type: int, sign: int = 1) -> float:
    """sign=+1 なら大きいほど左。sign=-1 なら大きいほど右。"""
    base = NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))
    if sign < 0:
        return NORMALIZED_WIDTH - base
    return base


def center_tiebreak(x: float) -> float:
    """同点をほどくためだけの中央寄り減点。端の手から先に落ちる。"""
    return CENTER_TIEBREAK_WEIGHT * abs(x - NORMALIZED_WIDTH / 2)


def wall_gap(fruit: Fruit, sign: int) -> float:
    """大側 (sign) の壁との隙間。"""
    if sign > 0:
        return fruit.x - fruit.radius
    return NORMALIZED_WIDTH - fruit.radius - fruit.x


def is_wall_anchored(fruit: Fruit, sign: int) -> bool:
    """大側の壁に付いているか。"""
    limit = max(EDGE_ANCHOR_MIN, fruit.radius * EDGE_ANCHOR_FRAC)
    return wall_gap(fruit, sign) <= limit


def _straight_fall_contact(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    held_r: float,
) -> Fruit | None:
    """列 x にまっすぐ落としたとき最初に触れる実。床だけなら None。

    弾かれて転がった後の実際の着地ではなく、狙った列にそのまま
    落ちた場合の幾何的な最初の接触相手 (物理は回さない)。
    """
    best: Fruit | None = None
    best_y = float(NORMALIZED_HEIGHT) - held_r  # 何にも触れなければ床。
    for fruit in fruits:
        dx = abs(fruit.x - x)
        gap = fruit.radius + held_r
        if dx >= gap:
            continue
        touch_y = fruit.y - math.sqrt(gap * gap - dx * dx)
        if touch_y < best_y:
            best_y = touch_y
            best = fruit
    return best


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


def _is_nestled(
    fruit: Fruit,
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> bool:
    """より大きい実どうしの谷に収まっているか。"""
    return _valley_flanks(fruits, fruit.x, fruit.type) is not None


def _size_order_exempt(
    fruit: Fruit,
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> bool:
    """大小順の対象から外す実か。

    谷にいるだけでは外さない。相方のいない実はその谷から出る当てが無く、
    ただの並び順違反として残る。外すと、逆転を作った実そのものが谷の壁として
    数えられ、自分が作った逆転を自分で免除する抜け穴になる (seed=49140 の
    9 手目: 梨とグレープの盤にデコポンをグレープの右へ置くと、そのデコポンが
    谷を成立させてグレープの逆転 1.5 が 0.14 に落ち、並べ直す手に 0.20 差で
    勝つ)。

    held と同種の谷はここからは見えないが、そちらは手ごとの `valley_grow_ok`
    が `VALLEY_GROW_BONUS` で拾うので、育成を潰すことにはならない。
    """
    if not _is_nestled(fruit, fruits):
        return False
    return any(f.type == fruit.type and f is not fruit for f in fruits)


def valley_grow_ok(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    drop_type: int,
    next_type: int | None,
) -> bool:
    """谷を育てにいく手か。基準は谷に入っている実で、壁の型は見ない。

    谷の実 (より大きい実に挟まれている実) を的にして、
    - その実が held と同種 → 落とせばそのまま合体する
    - その実が held よりひとつ大きく、held と next が同種 → 2 枚落とせば
      合体してその実と同じ型になり、更に合体する

    落下前の盤 (`before`) に対して呼ぶ。held はまだ盤に乗っていないので、
    的の実に自分自身が混ざることはない。
    """
    for fruit in fruits:
        if fruit.type == drop_type:
            pass
        elif (
            next_type is not None
            and drop_type == next_type
            and fruit.type == drop_type + 1
        ):
            pass
        else:
            continue
        # 谷は的の実を基準に取る。held を基準にすると、的の実そのものが
        # 「より大きい実」として壁側に回ってしまう (いちごから見た谷のぶどう)。
        flanks = _valley_flanks(fruits, fruit.x, fruit.type)
        if flanks is None:
            continue
        left, right = flanks
        if left.x < land_x < right.x:
            return True
    return False


# --- 減点項 ---------------------------------------------------------------


def board_penalties(
    fruits: list[Fruit], *, sign: int = 1, exempt_size_order: bool = False
) -> float:
    """落としたあとの盤面減点（埋め込み・肩乗り・同種過多・サイズ順・大寄せ）。

    exempt_size_order: held がこの手で合体したとき True。合体の反動で
    弾かれた無関係の実まで大小順違反として減点しない (`policy._evaluate_drop` 参照)。
    """
    penalty = 0.0
    penalty += BURY_WEIGHT * _bury_penalty(fruits)
    penalty += PERCH_WEIGHT * _perch_penalty(fruits)
    penalty += _excess_same_penalty(fruits)
    if not exempt_size_order:
        penalty += _size_order_penalty(fruits, sign)
    penalty += _big_layout_penalty(fruits, sign)
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
    big_min = max(0, max_t - BIG_CLUSTER_SPAN)
    large_left = sign > 0

    penalty = 0.0
    max_fruits = [fruit for fruit in fruits if fruit.type == max_t]

    for big in max_fruits:
        if not is_wall_anchored(big, sign):
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
        # 間の段を育てる場所は空けておく。ここを詰めると、あとで挟まる型が
        # ツモれたとき置き場が無く、外側へ回して並び順を壊すしかなくなる
        # (実測: 初手 orange の真横に grape を寄せると、次の dekopon が
        # grape の外側に落ちて 4-2-3 の並びになる)。空けるのは「すぐ次に
        # 要る 1 個ぶん」= 抜けている型のうち一番大きいものの直径だけで、
        # それを超えた余分は従来どおり減点する。
        missing = range(min(left.type, right.type) + 1, max(left.type, right.type))
        want = 2.0 * max((fruit_radius(t) for t in missing), default=0.0)
        gap -= want
        if gap <= 0:
            continue
        gap = min(gap, left.radius + right.radius)
        size = 0.5 + 0.05 * (left.type + right.type)
        penalty += cluster_weight * gap * size
    return penalty


def _excess_same_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """同種が 3 個以上ある超過分を減点。2 個までは合成待ちとして許容。

    超過分を累乗的に効かせる形は測って見送った
    (NOTES「終盤の低段位散在による即死」の改善試行)。
    """
    excess_same_weight = 20.0
    counts: dict[int, int] = {}
    for fruit in fruits:
        counts[fruit.type] = counts.get(fruit.type, 0) + 1
    penalty = 0.0
    for count in counts.values():
        if count >= 3:
            penalty += (count - 2) * excess_same_weight
    return penalty


def _size_order_penalty(fruits: list[Fruit], sign: int = 1) -> float:
    """左右の大小が逆転しているペアを減点。絶対 ideal より相対順を見る。

    より大きい実の谷にはまっていて、かつ盤に同種の相方が残っている実だけ
    大小順の対象外 (谷育成をレイアウト減点で潰さない)。条件は `_size_order_exempt`。
    """
    if not fruits:
        return 0.0
    size_order_pair_weight = 1.5
    size_order_ideal_weight = 0.004
    penalty = 0.0
    # _size_order_exempt は 1 個あたり O(n)。ペアごとに引き直すと O(n^3) に
    # なるので先に 1 回だけ引く。候補ごとに毎回走る場所。
    exempt = [_size_order_exempt(f, fruits) for f in fruits]
    open_fruits = [f for f, skip in zip(fruits, exempt) if not skip]
    for i, a in enumerate(fruits):
        if exempt[i]:
            continue
        for j in range(i + 1, len(fruits)):
            b = fruits[j]
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            if exempt[j]:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            if sign > 0 and left.type < right.type:
                penalty += (right.type - left.type) * size_order_pair_weight
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * size_order_pair_weight
    if open_fruits:
        penalty += (
            sum(abs(f.x - ideal_x(f.type, sign)) for f in open_fruits)
            / len(open_fruits)
            * size_order_ideal_weight
        )
    return penalty


def _bury_penalty(fruits: list[Fruit]) -> float:
    """合成候補を異種で埋める度合い。"""
    penalty = 0.0
    for under in fruits:
        for over in fruits:
            if over is under or over.type <= under.type:
                continue
            if over.y >= under.y:
                continue
            # 接触の窓は両方の半径で取る。下の実だけを基準にすると、
            # 大きい実が小さい実に乗るほど窓が狭まり、いちばん潰したい
            # 「大で小を埋める」形が検出から外れる。
            if abs(over.x - under.x) > (under.radius + over.radius) * 0.9:
                continue
            gap = (under.y - under.radius) - (over.y + over.radius)
            if -MERGE_SLACK <= gap <= under.radius * 0.6:
                siblings = sum(1 for f in fruits if f.type == under.type and f is not under)
                if siblings >= 1:
                    penalty += 1.0
                else:
                    penalty += 0.35
    return penalty


def _perch_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """大実の肩・頭に載った小実の減点。型差が大きいほど重い。

    `_bury_penalty` の裏返し。あちらは「小実の上の異種」を数えるので
    `over.type > under.type` の側しか見ず、逆向き (大実の上の小実) は
    どの規則からも漏れていた。横の `_size_order_penalty` も、縦に重なった
    ペアは列が同じとして除外し、谷にはまった実は `_size_order_exempt` で
    外すので、ここは素通りだった。

    大実の上面は次の段を作る場所で、型差の大きい実を載せるとその実は
    相方に会えないまま居座り、下の大実の合体面も塞ぐ。載っている判定は
    接触ではなく「大実の footprint の中で、下端が大実の中心より上」。
    直接触れていなくても、間に 1 段挟んで山の上に乗っている形を拾う。

    許した型差を 1 超えるごとに 1.0 を返す。重みは呼び元で掛ける。
    """
    if not fruits:
        return 0.0
    max_t = max(fruit.type for fruit in fruits)
    big_min = max_t - PERCH_BIG_SPAN
    penalty = 0.0
    for under in fruits:
        if under.type < big_min:
            continue
        for over in fruits:
            gap_type = under.type - over.type
            if gap_type < PERCH_MIN_GAP:
                continue
            if over.y + over.radius > under.y:
                continue
            if abs(over.x - under.x) > under.radius + over.radius:
                continue
            penalty += float(gap_type - PERCH_MIN_GAP + 1)
    return penalty


def foreign_aim_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    x: float,
    drop_type: int,
    held_r: float,
) -> float:
    """狙った列 x が異種のガチ真上かどうかの減点。

    真下が同種なら合体待ちで 0。肩着地や床着地も 0。
    """
    under = _straight_fall_contact(fruits, x, held_r)
    if under is None or under.type == drop_type:
        return 0.0
    # 中心がずれていれば真上ではない。肩着地は対象外。
    if abs(x - under.x) > under.radius * FOREIGN_AIM_CENTER_FRAC:
        return 0.0
    return FOREIGN_AIM_PENALTY

