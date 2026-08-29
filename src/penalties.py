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
FOREIGN_AIM_WEIGHT = 100.0
# 同点候補の順位を決めるためだけの重み。他の項が全部並ぶ局面は常態で、これが無いと
# 勝つ手が候補 set の列挙順 = float のハッシュ順という実装詳細で決まってしまう。
# 最小の合成点が 1.0 (cherry -> straw) なので、本物の差を覆さないよう最大でも
# その 5 分の 1 に収める (`|x - 中央|` は高々 190 なので 0.001 で 0.19)。
# 手の良し悪しをこれで表そうとしないこと。
CENTER_TIEBREAK_WEIGHT = 0.001

# 大側へ寄せた合体のボーナス (減点から引く形で入れる)。`merge_big_side_bonus`
# が返す値。合体どうしの順位を本家点で決める性質は壊さない。最小の
# 合成点差が 1.0 (cherry -> straw) なので、それを覆さない値に収める。
# 同点帯の幅 0.1 の 5 倍。
MERGE_BIG_SIDE_WEIGHT = 0.5
# 寄せたと認める最小の移動量 (落とした実の半径に対する割合)。合体位置は 2 中心の
# 中点なので、寄せる意図の無い手でも半径未満はふつうにずれる。
MERGE_BIG_SIDE_SLACK_FRAC = 1.0

# 相方から遮られた着地 1 手ぶん。`blocked_partner_penalty` が返す二値。
# 相方が盤にいるのに会えない位置へ自分から入る手を止める規則で、掛かるか否かの
# フィルタとして働かせる (効いている 3 本 perch/bury/foreign_aim がどれも二値
# なのと同じ形。NOTES「既存の重みには梃子が無い」)。値は同じ「相方の取り逃し」を
# 見る `BURY_WEIGHT` に合わせてある。
BLOCKED_PARTNER_WEIGHT = 20.0

# 谷育成ボーナス (減点から引く形で入れる)。`valley_grow_bonus` が返す値。
# 実合体より強くしない。8.0 だと grape 合体 (6 点) を蹴って非合体の谷を選んだ。
# 2.0 で育成側に倒れ、3.0 でも合体は取り続ける (実測)。
VALLEY_GROW_WEIGHT = 3.0

# --- 壁付き判定 (角ポケット減点と梯子の土台で共有) ---
EDGE_ANCHOR_MIN = 24.0
EDGE_ANCHOR_FRAC = 0.35

# --- 盤面減点の重み ---
# compare_policy の A/B がモジュール属性として差し替えるので、
# board_penalties のローカルではなくここに置く。
# 埋めた 1 個ぶん。相方が盤にいる実は次の手で合体できたので重い。
BURY_WEIGHT = 20.0
# 相方が盤にいない実を埋めた 1 個ぶん。相方はこれから引くので、屋根を掛けると
# その実は誰にも会えなくなる (居座る実の 46.6% が頭上を塞がれている。
# NOTES「居座る実 (fossil) の測定」)。相方待ちを潰すより軽いが、無視はしない。
BURY_LONE_WEIGHT = 15.0
# 大実の肩に載せてよい型差の上限。パイン (8) の肩にオレンジ (4) までは許す。
PERCH_MIN_GAP = 5
# 肩を見る実の範囲 (最大実から何段下まで)。0 なら最大実だけ。
PERCH_BIG_SPAN = 1
PERCH_WEIGHT = 16.0
# 肩乗りを免除する窪みの深さ。壁 (窪みの左右のうち小さいほう) との型差が
# これ以下なら、そこは次の段。相方が来ればその場で合体し、続けて壁とも合体できる。
# 3 seed の全候補で測ると、免除されるのは perch の 6.8% だけ。窪みなら免除、
# にすると 77.5% が消えるうえ、この規則を入れる動機になった局面のさくらんぼ自身が
# りんごとオレンジの窪みに載っているので、症例ごと免除してしまう。
PERCH_RUNG_MAX_GAP = 1
# 谷底に沈んだ実を数える壁との型差。1 段上の壁なら次の段なので数えない
# (`_is_rung` と同じ基準)。
PIT_MIN_GAP = 2
# 谷底 1 段ぶん。`_perch_penalty` より軽い。肩に載った実は上から相方が来れば
# 会えるが、谷底は壁が高いだけで上は開いているので、取り返しは効く。
PIT_WEIGHT = 8.0
# 同種 3 個目以降 1 個ぶん。
EXCESS_SAME_WEIGHT = 20.0
# 左右の大小逆転 1 段ぶん。
SIZE_ORDER_PAIR_WEIGHT = 1.5
# ideal_x からの平均乖離に掛ける。
SIZE_ORDER_IDEAL_WEIGHT = 0.004



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

    **相方は同じ谷の中にいなければならない。** 盤のどこかに 1 個あればよいと
    すると、谷の外の相方が届かないまま免除だけが立つ。谷を作っているのは自分より
    大きい実なので、その向こう側の相方とは合体できず、その実はただ並びを崩した
    まま居座る (seed=834761 の 35 手目: ナシとパインの谷に残ったいちごが、
    反対端 x=23 のいちごを根拠に免除され、逆転 7.5 が 0 になっていた)。

    held と同種の谷はここからは見えないが、そちらは手ごとの `valley_grow_bonus`
    が拾うので、育成を潰すことにはならない。
    """
    flanks = _valley_flanks(fruits, fruit.x, fruit.type)
    if flanks is None:
        return False
    left, right = flanks
    return any(
        f.type == fruit.type and f is not fruit and left.x < f.x < right.x
        for f in fruits
    )


def valley_grow_bonus(
    fruits: list[Fruit] | tuple[Fruit, ...],
    land_x: float,
    drop_type: int,
    next_type: int | None,
) -> float:
    """谷を育てにいく手へのボーナス。基準は谷に入っている実で、壁の型は見ない。

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
            return VALLEY_GROW_WEIGHT
    return 0.0


def merge_big_side_bonus(
    drop_x: float, held_fruit: Fruit | None, held_r: float, sign: int
) -> float:
    """合体でできた実が、落とした列より大側へ半径 1 つぶん以上寄った手へのボーナス。

    `held_fruit` は落とした実の系譜が最後に居る実 (`simulate_drop_held`)。合体は
    反動で新実を横へ飛ばすので、同じ相方に当てても左右どちらから当てるかで
    でき上がる並びが変わる。転がってから当てた場合も込みで、結果として
    大側 (`sign`) へ寄った当て方を選ばせる。

    合体した手にだけ掛ける。合体しなかった手の着地は `_size_order_penalty` が
    そのまま見ているので、こちらで二重に見ない。
    """
    if held_fruit is None:
        return 0.0
    toward_big = -sign * (held_fruit.x - drop_x)
    if toward_big < MERGE_BIG_SIDE_SLACK_FRAC * held_r:
        return 0.0
    return MERGE_BIG_SIDE_WEIGHT


def _partner_blocked(a: Fruit, b: Fruit, fruits: list[Fruit] | tuple[Fruit, ...]) -> bool:
    """同種 a と b の間が、自分より大きい実で遮られているか。

    遮る条件は「型差が `PIT_MIN_GAP` 以上」「横に挟まる」「頭が両者の中心より上」
    の 3 つ。何段も大きい実は押しても合体でも動かせないので、頭が中心より上に
    出ていれば越えて相方に届く経路が無い。同型・小型は押し出せるし合体で消える。

    **1 段上は壁に数えない。** そこは次の段で、相方が来ればその場で合体して
    壁に追いつける (`_pit_penalty` の `PIT_MIN_GAP`、`_is_rung` の
    `PERCH_RUNG_MAX_GAP` と同じ線引き)。1 段上まで壁にすると、段の窪みへ置く
    正しい手が相方から遮られた扱いになり、小実に屋根を掛ける側へ倒れる
    (tests/test_policy.py の `test_uses_the_next_rung_instead_of_roofing_a_small_fruit`)。
    """
    lo, hi = (a.x, b.x) if a.x <= b.x else (b.x, a.x)
    top = min(a.y, b.y)
    for wall in fruits:
        if wall.type - a.type < PIT_MIN_GAP:
            continue
        if not lo < wall.x < hi:
            continue
        if wall.y - wall.radius < top:
            return True
    return False


def blocked_partner_penalty(
    fruits: list[Fruit] | tuple[Fruit, ...],
    held_fruit: Fruit | None,
) -> float:
    """落とした実が、盤にいる相方のどれからも遮られる位置へ入った手の減点。

    居座る実の測定で、50 手以上残る実は **66.1% の手で相方が盤にいた**
    (NOTES「居座る実 (fossil) の測定」)。合体待ちではなく取り逃しで、供給側の
    穴として名指しされていた形。`_bury_penalty` は屋根を、`_pit_penalty` は
    左右とも大実に挟まれた谷を見るが、**片側が壁でもう片側だけが大実**という
    形はどちらにも掛からない (`_valley_flanks` は左右そろって初めて谷とみなす)。
    `_corner_pocket_penalty` が拾うのは最大実が大側壁に付いた角だけ。

    盤全体の到達可能性を合計しない。今の手で動かせない盤の性質はどの候補でも
    同じ値になり、順位を変えない (NOTES「今の手で動かせない盤の性質には減点を
    付けない」)。見るのは**落とした実自身が入った位置**だけで、候補ごとに割れる。

    相方が 1 個でも届くなら 0。全部遮られて初めて掛かる二値。相方がそもそも
    盤にいない実は取り逃しではないので対象外 (そちらは `BURY_LONE_WEIGHT`)。

    合体しなかった手にだけ掛ける (呼び元)。合体した手は本家点が付くうえ、
    できた実は 1 段上の別型なので、この規則で合体の動機を削らない。
    """
    if held_fruit is None:
        return 0.0
    # `_lineage_fruit` は after と同じ値を持つ別インスタンスを返すので、
    # 自分自身を外すのは `is` ではなく位置で見る (simulate_drop_held の docstring)。
    partners = [
        f
        for f in fruits
        if f.type == held_fruit.type
        and not (f.x == held_fruit.x and f.y == held_fruit.y)
    ]
    if not partners:
        return 0.0
    if any(not _partner_blocked(held_fruit, p, fruits) for p in partners):
        return 0.0
    return BLOCKED_PARTNER_WEIGHT


# --- 減点項 ---------------------------------------------------------------


def board_penalties(
    fruits: list[Fruit], *, sign: int = 1, exempt_size_order: bool = False
) -> float:
    """落としたあとの盤面減点（埋め込み・肩乗り・同種過多・サイズ順・角ポケット）。

    exempt_size_order: held がこの手で合体したとき True。合体の反動で
    弾かれた無関係の実まで大小順違反として減点しない (`policy._evaluate_drop` 参照)。
    """
    penalty = 0.0
    penalty += _bury_penalty(fruits)
    penalty += PERCH_WEIGHT * _perch_penalty(fruits)
    penalty += PIT_WEIGHT * _pit_penalty(fruits)
    penalty += _excess_same_penalty(fruits)
    if not exempt_size_order:
        penalty += _size_order_penalty(fruits, sign)
    penalty += _corner_pocket_penalty(fruits, sign)
    return penalty


def _corner_pocket_penalty(fruits: list[Fruit] | tuple[Fruit, ...], sign: int = 1) -> float:
    """大側端の角ポケット減点。

    sign=+1 なら左が大側、-1 なら右が大側。角ポケットはその側だけ見る。
    最大実 L が大側壁に付いているとき、L より外側かつ L.y より下の小実を強く減点。
    L の裏に入った実は合体の相方に会えないまま居座る。
    """
    if not fruits:
        return 0.0
    max_t = max(fruit.type for fruit in fruits)

    under_l_weight = 50.0
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

    return penalty


def _excess_same_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """同種が 3 個以上ある超過分を減点。2 個までは合成待ちとして許容。

    超過分を累乗的に効かせる形は測って見送った
    (NOTES「終盤の低段位散在による即死」の改善試行)。
    """
    counts: dict[int, int] = {}
    for fruit in fruits:
        counts[fruit.type] = counts.get(fruit.type, 0) + 1
    penalty = 0.0
    for count in counts.values():
        if count >= 3:
            penalty += (count - 2) * EXCESS_SAME_WEIGHT
    return penalty


def _size_order_penalty(fruits: list[Fruit], sign: int = 1) -> float:
    """左右の大小が逆転しているペアを減点。絶対 ideal より相対順を見る。

    より大きい実の谷にはまっていて、かつ盤に同種の相方が残っている実だけ
    大小順の対象外 (谷育成をレイアウト減点で潰さない)。条件は `_size_order_exempt`。
    """
    if not fruits:
        return 0.0
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
                penalty += (right.type - left.type) * SIZE_ORDER_PAIR_WEIGHT
            elif sign < 0 and left.type > right.type:
                penalty += (left.type - right.type) * SIZE_ORDER_PAIR_WEIGHT
    if open_fruits:
        penalty += (
            sum(abs(f.x - ideal_x(f.type, sign)) for f in open_fruits)
            / len(open_fruits)
            * SIZE_ORDER_IDEAL_WEIGHT
        )
    return penalty


def _bury_counts(fruits: list[Fruit] | tuple[Fruit, ...]) -> tuple[float, float]:
    """異種で埋めた実の数を (相方あり, 相方なし) に分けて返す。

    重みが 2 つある規則なので、`band_escape.py` が別々に掃引できるよう
    数える側と重みを分けてある (AGENTS「1 つの項に 1 つの規則」)。
    """
    paired = lone = 0.0
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
                    paired += 1.0
                else:
                    lone += 1.0
    return paired, lone


def _bury_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """合成候補を異種で埋める減点。相方の有無で重みが違う。"""
    paired, lone = _bury_counts(fruits)
    return BURY_WEIGHT * paired + BURY_LONE_WEIGHT * lone


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
    # 段かどうかは載っている実だけで決まるので、下の実ごとに引き直さない。
    rung: dict[int, bool] = {}
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
            if id(over) not in rung:
                rung[id(over)] = _is_rung(over, fruits)
            if rung[id(over)]:
                continue
            penalty += float(gap_type - PERCH_MIN_GAP + 1)
    return penalty


def _pit_penalty(fruits: list[Fruit] | tuple[Fruit, ...]) -> float:
    """自分よりずっと大きい実の谷底にいて、同じ谷に相方がいない実の減点。

    `_perch_penalty` が大実の「上」を見るのに対し、こちらは「間」。壁が自分より
    何段も大きいと横へは合体できないので、相方は真上の狭い隙間から来るしかない。
    `_size_order_exempt` の docstring が名指ししている「その谷から出る当てが無い
    実」そのもので、これまで値付けは `_size_order_penalty` のペア差 1.5 しか
    無かった。大域のペア数は 1 個の置き場でほとんど動かないので効かない
    (NOTES「効かなかった案」)。ここは落とした実自身が作る形なので候補間で割れる。

    壁との型差 1 段は次の段なので数えない (`PIT_MIN_GAP`)。同じ谷に同種の相方が
    いるなら育成中とみなして外す。判定は `_size_order_exempt` と揃える。

    型差が `PIT_MIN_GAP` を 1 超えるごとに 1.0 を返す。重みは呼び元で掛ける。
    """
    penalty = 0.0
    for fruit in fruits:
        flanks = _valley_flanks(fruits, fruit.x, fruit.type)
        if flanks is None:
            continue
        left, right = flanks
        gap = min(left.type, right.type) - fruit.type
        if gap < PIT_MIN_GAP:
            continue
        if any(
            f.type == fruit.type and f is not fruit and left.x < f.x < right.x
            for f in fruits
        ):
            continue
        penalty += float(gap - PIT_MIN_GAP + 1)
    return penalty


def _is_rung(fruit: Fruit, fruits: list[Fruit] | tuple[Fruit, ...]) -> bool:
    """次の段の窪みに収まっているか。大実の裸の上面と区別する。

    小さい実は、盤が大実で埋まると**どの肩に置いても型差が開く**。逃げ場が
    無くなると、方策は肩を避けて別の小実に屋根を掛ける側へ倒れる (seed=890270
    の 72 手目: グレープを桃の肩に置くと 16、パインなら 32、いちごに屋根を
    掛けると 15 で、屋根がいちばん安かった)。屋根を掛けられた実は上から相方が
    届かないので、肩より重いはずのものが軽くなっていた。

    ただし窪みならどこでもよいわけではない。`_perch_penalty` を入れる動機に
    なった局面のさくらんぼも、りんごとオレンジの窪みに載っていた。分けるのは
    **壁との型差**で、1 段上の壁 (`PERCH_RUNG_MAX_GAP`) なら相方が来た時点で
    合体して壁に追いつける。
    """
    flanks = _valley_flanks(fruits, fruit.x, fruit.type)
    if flanks is None:
        return False
    left, right = flanks
    return min(left.type, right.type) - fruit.type <= PERCH_RUNG_MAX_GAP


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
    return FOREIGN_AIM_WEIGHT

