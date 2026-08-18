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
import statistics

from .observe import clamp_drop_x
from .sim.sim_physics import landed_xy, simulate_drop_held
from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE, SPAWN_MAX_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# --- 複数箇所で共有するチューニング ---
# 合成できそうな接触の許容 (中心距離と半径和の差)。
MERGE_SLACK = 18.0
# 異種真上とみなす着地の横ずれ (下実半径に対する割合)。
FOREIGN_AIM_CENTER_FRAC = 0.20
# 真下の異種中心帯に着地したときの減点。
FOREIGN_AIM_PENALTY = 100.0
# 谷育成ボーナス (減点から引く形で入れる)。`valley_grow_ok` が成立する着地だけ。
# 実合体より強くしない。8.0 だと grape 合体 (6 点) を蹴って非合体の谷を選んだ。
# 2.0 で育成側に倒れ、3.0 でも合体は取り続ける (実測)。
VALLEY_GROW_BONUS = 3.0

# --- 壁付き判定 (角ポケット減点と梯子の土台で共有) ---
EDGE_ANCHOR_MIN = 24.0
EDGE_ANCHOR_FRAC = 0.35

# 「大きい実の塊」とみなす最大実からの段数。
BIG_CLUSTER_SPAN = 2

# --- 縦の大小順 ---
# 縦に積まれているとみなす横ずれ (半径和に対する割合)。1.0 で横に並んだ状態
# なので、それより締める。肩に載せた形はここに入るが、肩は上が小さい側なので
# 減点にはならない (梯子の 4→5→6→7 は素通りする)。
VERTICAL_STACK_FRAC = 0.8
# 上下が入れ替わっているとみなす最小の高さ差 (半径和に対する割合)。
# ほぼ同じ高さで横に並んだだけの組を「積んである」と読まないための下限。
VERTICAL_STACK_MIN_RISE = 0.35
# 上が大きいペア 1 組の減点。型差に掛ける。
# 3.0 にすると縦を守るために横を崩し始める (166 局面で横の逆転が +0.15 に転じる)。
# 0.75 は縦の効きが 1.5 と変わらないぶん横が弱い。
VERTICAL_ORDER_WEIGHT = 1.5

# --- 閉じ込め ---
# より大きい実に左右から挟まれて、出る当ての無い実の減点。1 個あたり。
# 谷育成 (VALLEY_GROW_BONUS) の裏返しで、育てられる谷は褒め、育たない谷は罰する。
# 相方が盤に残っている実は掛けない。そちらは合体待ちであって閉じ込めではなく、
# 罰すると谷に餌を入れる手そのものを潰す (実測: 相方ありも 1.0 で数えたところ、
# seed=74546 で中盤の連鎖が起きなくなり 241 手 -> 163 手に落ちた。step 140 の
# 実数が 9 個 -> 22 個、頭頂が 228 -> 45)。切り分けは `_size_order_exempt` と同じ。
# 230 局面のスクリーニング: 重みを上げるほど閉じ込めも逆転も減り、合体回数は
# ほぼ不変 (1.0/2.0/4.0/8.0 で合体 235/235/234/233)。8.0 は逆転が反転して
# (+26 -> +28) 合体も落ちるので 4.0 が折り返し。一致率は 93.0%。
TRAPPED_WEIGHT = 4.0

# --- 段階の切り分け ---
# 盤が「崩れている」とみなす逆転率。横の大小順が逆転しているペアの割合で、
# 0.5 は完全に無秩序 (向きに情報が無い) を意味する。これを超えた盤でだけ、
# 立て直し側の規則 (谷育成) を掛ける。整った盤でそれを掛けると、
# 小さい実を小側へ置く手を潰して盤を崩し始める (実測: cherry は汚さない手が
# 候補にある 97% の局面のうち 64% でしかそれを選べていなかった)。
# 0.25 では一度も閉じない。実測の逆転率は中央 0.333 で、序盤 30 手が 0.156、
# 90 手以降は 0.35〜0.39。0.25 に置くと中盤以降ずっと「崩れている」側に落ちて、
# 一致率も選ぶ手もゲート無しと 1 桁まで完全に一致した (166 局面)。
BROKEN_INVERSION_FRAC = 0.35

# --- 床の埋まり具合 ---
# 床に着いているとみなす高さ。半径のこの倍率ぶん下端に寄っていれば床置き。
FLOOR_BAND = 1.35
# 埋まっているとみなす隙間の上限 = オレンジの直径。
# 壁から壁まで繋がっている必要はなく、オレンジが収まらない隙間なら
# そこへ落とす手が問題にならないので埋まり扱いにする。
FLOOR_PACKED_GAP = fruit_radius(SPAWN_MAX_TYPE) * 2.0

# --- 床埋め後の大ツモ ---
# 床が埋まると小さい側に置く場所が無くなる。それでも ideal_x は小さい実を
# 小側へ引き続けるので (orange の ideal は 236 = 右寄り)、大きめのツモを
# 小側の上に積んで下の小実を潰し、崩れる。床が埋まったら横並びではなく
# 大側の肩へ載せる。梯子の形はこの置き分けの結果として出てくる。
# 実測 (10 シード×120 手): 該当 358 回のうち 211 回を小側へ置き、その 210 回で
# eval が本気で小側を選んでいた (中央値 +4.1)。候補ではなく評価の問題。
PACKED_BIG_DRAW_MIN_TYPE = SPAWN_MAX_TYPE - 1
# 中央値 +4.1 の僅差をひっくり返しつつ、小側で実際に合成できる手 (最大 +159.9)
# は残す。合成する手には掛けない (merges == 0 のときだけ) ので合成とは争わない。
PACKED_SMALL_SIDE_WEIGHT = 8.0


# --- 盤面を読むだけの補助 -------------------------------------------------


def ideal_x(fruit_type: int, sign: int = 1) -> float:
    """sign=+1 なら大きいほど左。sign=-1 なら大きいほど右。"""
    base = NORMALIZED_WIDTH * (1.0 - (fruit_type + 0.5) / (MAX_FRUIT_TYPE + 1))
    if sign < 0:
        return NORMALIZED_WIDTH - base
    return base


def wall_gap(fruit: Fruit, sign: int) -> float:
    """大側 (sign) の壁との隙間。"""
    if sign > 0:
        return fruit.x - fruit.radius
    return NORMALIZED_WIDTH - fruit.radius - fruit.x


def is_wall_anchored(fruit: Fruit, sign: int) -> bool:
    """大側の壁に付いているか。"""
    limit = max(EDGE_ANCHOR_MIN, fruit.radius * EDGE_ANCHOR_FRAC)
    return wall_gap(fruit, sign) <= limit


def _top_crown(fruits: list[Fruit]) -> float:
    """一番上の頭頂 y。空なら床。"""
    if not fruits:
        return float(NORMALIZED_HEIGHT)
    return min(f.y - f.radius for f in fruits)


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


def _typed_pairs(
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> "list[tuple[Fruit, Fruit]]":
    """型の違う組。同種どうしは合体待ちなので大小順を問わない。

    横に重なった組を外すかどうかは向きで変わるので、ここではしない。
    横は左右を言えない組を外し (`inversion_fraction`)、縦はその組こそが
    本体 (`_vertical_order_penalty`)。
    """
    return [
        (a, b)
        for i, a in enumerate(fruits)
        for b in fruits[i + 1 :]
        if a.type != b.type
    ]


def _floor_row(fruits: list[Fruit] | tuple[Fruit, ...]) -> list[Fruit]:
    """床に着いている実を x 順で。"""
    return sorted(
        (f for f in fruits if f.y > NORMALIZED_HEIGHT - f.radius * FLOOR_BAND),
        key=lambda f: f.x,
    )


def _floor_gap(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """床の一番広い隙間。壁との隙間も数える。空なら盤幅。"""
    row = _floor_row(fruits)
    if not row:
        return float(NORMALIZED_WIDTH)
    worst = max(
        row[0].x - row[0].radius,
        NORMALIZED_WIDTH - (row[-1].x + row[-1].radius),
    )
    for left, right in zip(row, row[1:]):
        worst = max(worst, (right.x - right.radius) - (left.x + left.radius))
    return worst


def _floor_packed(fruits: list[Fruit] | tuple[Fruit, ...]) -> bool:
    """床が埋まっているか。

    壁から壁まで繋がっている必要はない。隙間がオレンジの直径以下なら、
    そこへ落としても問題にならないので埋まっているとみなす。
    """
    return _floor_gap(fruits) <= FLOOR_PACKED_GAP


def _big_cluster_edge(
    fruits: list[Fruit] | tuple[Fruit, ...],
    max_type: int,
    sign: int,
) -> float:
    """大きい実の塊の、小さい側の縁。

    最大実 1 個ではなく、その 2 段下までを塊として見る (_big_layout_penalty と
    同じ括り)。最大実だけで切ると、隣の梨やリンゴまで小側扱いになる。
    """
    big_min = max(0, max_type - BIG_CLUSTER_SPAN)
    bigs = [fruit for fruit in fruits if fruit.type >= big_min]
    if sign > 0:
        return max(fruit.x + fruit.radius for fruit in bigs)
    return min(fruit.x - fruit.radius for fruit in bigs)


def _widest_gap(
    fruits: list[Fruit] | tuple[Fruit, ...],
    lo: float,
    hi: float,
) -> tuple[float, float]:
    """[lo, hi] の床にある一番広い隙間の幅と中心 x。"""
    widest = 0.0
    center = (lo + hi) / 2.0
    cursor = lo
    for fruit in _floor_row(fruits):
        if fruit.x + fruit.radius <= lo or fruit.x - fruit.radius >= hi:
            continue
        gap = (fruit.x - fruit.radius) - cursor
        if gap > widest:
            widest = gap
            center = (cursor + (fruit.x - fruit.radius)) / 2.0
        cursor = max(cursor, fruit.x + fruit.radius)
    gap = hi - cursor
    if gap > widest:
        widest = gap
        center = (cursor + hi) / 2.0
    return widest, center


def _small_side_room_ok(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    held_r: float,
    max_type: int,
    sign: int,
) -> bool:
    """小さい側に実際に落として、床に着地できるか。

    床の隙間を幾何だけで測ると、隙間の上に別の実がせり出していて実際には
    入らないケースを見逃す (指摘を受けて修正)。幅が物理的に入りそうな
    ときだけ、一番広い隙間の中心へ実際に 1 回落として確かめる。
    """
    edge = _big_cluster_edge(fruits, max_type, sign)
    lo, hi = (edge, float(NORMALIZED_WIDTH)) if sign > 0 else (0.0, edge)
    lo, hi = max(lo, held_r), min(hi, NORMALIZED_WIDTH - held_r)
    if lo > hi:
        return False
    widest, center = _widest_gap(fruits, lo, hi)
    if widest < held_r * 2.0:
        return False
    x = clamp_drop_x(center, drop_type)
    after, merges, _merge_types, held_merged = simulate_drop_held(fruits, drop_type, x)
    if merges > 0:
        return True
    land_x, land_y = landed_xy(fruits, after, drop_type, x, held_r, held_merged)
    floor_y = NORMALIZED_HEIGHT - held_r
    return land_y >= floor_y - 4.0 and lo - MERGE_SLACK <= land_x <= hi + MERGE_SLACK


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


def inversion_fraction(fruits: list[Fruit] | tuple[Fruit, ...], sign: int) -> float:
    """大小順を外しているペアの割合。0 = 整列、0.5 = 無秩序。

    `_size_order_penalty` と違って重みも ideal_x も見ない生の割合。段階を
    切り分けるゲート用で、減点の大きさではなく盤の状態そのものを表す。

    谷にいる実は、左右**どちらの壁に対しても**外していると数える。ペアの
    左右だけで数えると、より大きい実 2 つに挟まれた小さい実が片側としか
    逆転せず、明らかに崩れた盤が整列側に入ってしまう (オレンジ・ぶどう・
    りんご の並びで、ぶどうが逆転するのは オレンジ とだけなので 1/3 = 0.333。
    これは 0.35 を下回る)。谷込みで数えると 2/3 = 0.667 になる。

    `_size_order_exempt` が谷の実を減点から外すのと向きが逆に見えるが、
    別の仕事をしている。あちらは「育てる予定の実を二重に罰しない」ための
    免除で、こちらは「この盤は整っているか」の読み取り。
    """
    pairs = [
        (a, b)
        for a, b in _typed_pairs(fruits)
        if abs(a.x - b.x) >= min(a.radius, b.radius) * 0.5
    ]
    if not pairs:
        return 0.0
    flanks: dict[int, set[int]] = {}
    for fruit in fruits:
        walls = _valley_flanks(fruits, fruit.x, fruit.type)
        if walls is not None:
            flanks[id(fruit)] = {id(walls[0]), id(walls[1])}
    bad = 0
    for a, b in pairs:
        if id(b) in flanks.get(id(a), ()) or id(a) in flanks.get(id(b), ()):
            bad += 1
            continue
        left, right = (a, b) if a.x <= b.x else (b, a)
        if sign > 0 and left.type < right.type:
            bad += 1
        elif sign < 0 and left.type > right.type:
            bad += 1
    return bad / len(pairs)


def board_is_broken(fruits: list[Fruit] | tuple[Fruit, ...], sign: int) -> bool:
    """立て直し側の規則を掛けてよい盤か。整った盤では大小順を優先する。"""
    return inversion_fraction(fruits, sign) > BROKEN_INVERSION_FRAC


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
    """落としたあとの盤面減点（危険・埋め込み・同種過多・サイズ順・大寄せ・凸凹）。

    exempt_size_order: held がこの手で合体したとき True。合体の反動で
    弾かれた無関係の実まで大小順違反として減点しない (`policy._evaluate_drop` 参照)。
    横と縦の大小順はどちらも同じ反動を受けるので、まとめてこの旗で外す。
    """
    # 盤面が壁の内側基準になった分だけ、旧基準の 90.0 を座標変換してある。
    danger_y = 70.9
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
    # 閉じ込めは合体した手でも免除しない。大小順のペア数と違って反動で
    # 数がぶれる量ではなく、その手が残した盤の形そのものなので。
    penalty += _trapped_penalty(fruits)
    if not exempt_size_order:
        penalty += _size_order_penalty(fruits, sign)
        penalty += _vertical_order_penalty(fruits)
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

    三角数化 (超過分を累乗的に効かせる) を試したが、40 エピソードのペア比較
    (同シード, baseline 2143.57 -> 2079.05) で有意な改善なし。実測で追った
    死因 (低段位が合体先を失って散在 → 終盤に retreat 先が無くなる) は
    ペナルティの重みではなく、合体候補の実が大きい異種に側面から挟まれて
    物理的に合体不能になる配置の問題。手を打つならここではなく候補選び側。
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


def _trapped_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """より大きい実に左右から挟まれた実の減点。

    `valley_grow_ok` が「これから育てられる谷」を褒めるのに対し、こちらは
    「作ってしまった谷」を罰する。裏返しの規則が無かったので、盤を壊す手と
    壊さない手を eval がほとんど区別できていなかった。

    実測 (seed=74546 の 25 手目): 逆転率 0.000・閉じ込め 0 の完璧な盤で、
    いちごを閉じ込める手が、閉じ込めない 13 本の候補を **eval 0.07 差**で
    上回った (全候補が -6.12〜-6.21 の幅 0.09 に収まる同点プラトー)。
    その閉じ込めが 3 手後に置き場を失わせ、13 手後の桃-オレンジ-桃 につながる。

    掛けるのは**相方の残っていない実だけ**。相方がいれば合体して大きくなり
    谷から出られるので、それは合体待ちであって閉じ込めではない。
    """
    penalty = 0.0
    for fruit in fruits:
        if any(f.type == fruit.type and f is not fruit for f in fruits):
            continue
        if _valley_flanks(fruits, fruit.x, fruit.type) is None:
            continue
        penalty += TRAPPED_WEIGHT
    return penalty


def _vertical_order_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """上に大きい実が乗っている組を減点。大きいほど下という縦の並び。

    横の `_size_order_penalty` は x しか見ないので、縦の積み方には規則が
    無かった (実測: 縦に重なった 23079 組のうち 47% が逆さ＝ほぼ無秩序)。

    掛かるのは**上のほうが大きい**ときだけ。大きい実の肩に小さい実を載せる形は
    上が小さい側なので型差がいくつあっても 0 で、梯子 (桃の肩にオレンジ) は
    素通りする。逆に、小さい実の上に大きい実を乗せる手ほど型差ぶん重くなる。
    """
    penalty = 0.0
    for a, b in _typed_pairs(fruits):
        span = a.radius + b.radius
        if abs(a.x - b.x) > span * VERTICAL_STACK_FRAC:
            continue
        upper, lower = (a, b) if a.y < b.y else (b, a)
        # 横に並んだだけの組を「積んである」と読まない。
        if lower.y - upper.y < span * VERTICAL_STACK_MIN_RISE:
            continue
        if upper.type <= lower.type:
            continue
        penalty += (upper.type - lower.type) * VERTICAL_ORDER_WEIGHT
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


def packed_small_side_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    drop_type: int,
    held_r: float,
    sign: int,
) -> float:
    """床が埋まったあと、大きめのツモを小さい側へ逃がす減点。

    床が埋まると小側に横並びの場所は無い。そこへ置くと下の小実を潰して崩れる。
    最大実の内側の縁より小側に落ちたら減点し、大側の肩へ載せる手を選ばせる。
    合成する手には掛けない (呼び元が merges == 0 のときだけ呼ぶ)。
    """
    if drop_type < PACKED_BIG_DRAW_MIN_TYPE:
        return 0.0
    if not fruits or not _floor_packed(fruits):
        return 0.0
    max_type = max(fruit.type for fruit in fruits)
    # 自分と同じか小さい実しか無いなら、大側という概念が立たない。
    if max_type <= drop_type:
        return 0.0
    if (land_x - _big_cluster_edge(fruits, max_type, sign)) * sign <= 0.0:
        return 0.0
    # 小側にこのツモが素直に置けるなら、そこへ置くのが普通の手。
    # 大側へ回すのは「置くしかない」ときだけ。ここを一律の隙間幅で切ると
    # 中盤ずっと発火し、小側の生産ライン (orange->apple->pear) が枯れる。
    if _small_side_room_ok(fruits, drop_type, held_r, max_type, sign):
        return 0.0
    return PACKED_SMALL_SIDE_WEIGHT
