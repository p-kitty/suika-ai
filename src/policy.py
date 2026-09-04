"""落とす列を決める。薄い bootstrap 方策 (RL の土台)。

具体手順 (押し込み・復元押し・連鎖隙間空け・梯子発火など) は持たない。
合成・埋め込み・薄い大小順・転がり事故防止だけ見る。
手の採点は eval = score (本家の合成点) - penalties (事故・悪手の減点)。
死ぬ手だけは eval で競わせず、生きる手があるなら候補から外す (`choose_x`)。

減点側は `penalties.py`。ここは候補列の生成・1 手評価・next 先読みだけ。
`pen.X` の形で参照するのは、scripts/compare_policy.py の A/B が
モジュール属性を書き換えて重みを差し替えるため (import で束縛しない)。
"""

from __future__ import annotations

import itertools
import math
from concurrent.futures import Executor

from . import penalties as pen
from .observe import Observation, clamp_drop_x
from .reward import is_lost, merge_score
from .sim.sim_physics import landed_xy
from .sim.sim_physics import simulate_drop_held
from .vision.classify import fruit_radius
from .vision.colors import SPAWN_MAX_TYPE
from .vision.normalized import NORMALIZED_WIDTH
from .vision.state import Fruit

# --- 先読みと候補の粗さ ---
# next 手の割引。
NEXT_DISCOUNT = 0.55
# 探索の粗さ。物理 (simulate_drop) が支配的で、ここが実行時間をほぼ決める。
# 2/32 から広げた形。1 手のコストは 3.6 倍になるが、score・生存手数・スイカ到達が
# そろって伸びる (NOTES「採用: 先読みを 8/16 へ広げた」)。
# next 先読みを回す held 候補の本数。ここが効いている側で、2 のままだと
# 同点帯に並んだ候補のうち 2 本しか next で比べられない。
HELD_TOP = 8
# next 先読みの候補刻み。held (CANDIDATE_STEP) より粗い。
# ここを 32 に戻すと手の 96.2% は同じままコストが 3.6 → 2.6 倍に落ちる。
NEXT_CANDIDATE_STEP = 16.0
# held 候補の均等刻み。粗くすると危険な山の真上が候補に乗るので下げない
# (20 で test_avoids_dangerous_tall_stack が落ちた)。1 手の予算は先読み側
# (`HELD_TOP`) に寄せてあるので、ここを削って稼ぐ形にはなっていない。
# 細かくする側も打ち止め。転がって同種に届く窓は 1〜3px しか無いことがあり、
# 3.0 まで下げれば拾えるが 1 手 248 → 540ms に対し score は動かなかった
# (NOTES「候補の刻みと合体の窓」)。
CANDIDATE_STEP = 12.0


def _held_eval_job(
    obs: Observation, held_r: float, x: float
) -> tuple[float, float, list[Fruit], float]:
    """held 候補 1 個ぶんの (eval, x, after, 本家点)。プールに投げる単位。

    本家点を eval と別に返すのは、価値関数の学習が Q = 本家点 + V(after) の形で
    候補を並べるため (`training/collect.py`)。eval は score - penalties なので
    そこからは割り戻せない。
    """
    after, held_eval, score = _held_eval(obs, x, held_r)
    return held_eval, x, after, score


def rank_candidates(
    obs: Observation, *, pool: Executor | None = None
) -> list[tuple[float, float, list[Fruit], float]]:
    """候補ごとの (eval, x, 落下後の盤, 本家点) を eval 降順で返す。

    `choose_x` の 1 手目の評価そのもの。next 先読みの手前で切ってあるのは、
    学習データの収集が同じ物理を二度回さずに候補表を取れるようにするため
    (`training/collect.py`)。1 手のほぼ全部が `simulate_drop` なので、
    収集側で引き直すと素直に倍かかる (NOTES「実行コスト: 物理の高速化と探索幅」)。
    """
    if obs.held_type is None:
        raise ValueError("held_type が無い")

    held_r = fruit_radius(obs.held_type)
    xs = [
        clamp_drop_x(x, obs.held_type)
        for x in _candidates(list(obs.fruits), obs.held_type, held_r, extra_type=obs.next_type)
    ]
    if not xs:
        return []

    if pool is None:
        ranked = [_held_eval_job(obs, held_r, x) for x in xs]
    else:
        ranked = list(
            pool.map(_held_eval_job, itertools.repeat(obs), itertools.repeat(held_r), xs)
        )

    ranked.sort(key=lambda row: row[0], reverse=True)
    return ranked


def choose_x(
    obs: Observation,
    *,
    pool: Executor | None = None,
    ranked: list[tuple[float, float, list[Fruit], float]] | None = None,
) -> float:
    """観測から落とす列を返す。ready で held_type がある前提。

    pool を渡すと held/next 候補の simulate_drop をプロセスプールに分散する。
    結果は逐次実行と同じ (どの候補も互いに独立、盤面は読むだけ)。

    ranked に `rank_candidates` の結果を渡すと 1 手目の物理を引き直さない。
    候補表そのものが欲しい収集側 (`training/collect.py`) が、教師の手を
    ここで決め直させるために使う。手選びの規則を写して持たせない。
    """
    if ranked is None:
        ranked = rank_candidates(obs, pool=pool)
    if not ranked:
        return NORMALIZED_WIDTH / 2

    # 死ぬ手は eval で競わせない。減点で表すと、汚い盤を避けた分が死の重さを
    # 上回って自殺する (428 局面で 5 件。生きる手が 30〜45 本あるのに選んでいた)。
    # 差は最大 261 あったので、有限の減点では足りない。
    alive = [row for row in ranked if not is_lost(row[2])]
    if alive:
        ranked = alive
    if obs.next_type is None:
        return ranked[0][1]

    # next 先読みは held の eval 上位だけ (物理が重い)。候補は held より粗い刻み。
    best_x = ranked[0][1]
    best_score = -math.inf
    for held_eval, x, after, _score in ranked[:HELD_TOP]:
        value = held_eval + NEXT_DISCOUNT * _best_next_score(
            after, obs.next_type, step=NEXT_CANDIDATE_STEP, pool=pool
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
    # lo..hi に入る grid の倍数を全部並べる。
    xs = {i * grid for i in range(math.ceil(lo / grid), int(hi / grid) + 1)}
    xs.add(pen.ideal_x(drop_type, sign))
    _add_near_fruit_x(xs, fruits, held_r, lambda t: drop_type <= t <= drop_type + 2)

    if extra_type is not None:
        xs.add(pen.ideal_x(extra_type, sign))
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
    after, score, penalties, merges, held_merged = _evaluate_drop(
        before,
        drop_type,
        clamp_drop_x(x, drop_type),
        held_r,
        next_type=next_type,
    )
    # 落下前ぶんは落下後と同じ基準で引く。after 側で size_order を除外したのに
    # before 側で込みのまま引くと、存在しない減点を差し引いて merge 手が
    # 不当に有利になる (実測 0.303 のずれ)。
    penalties -= pen.board_penalties(
        before, sign=_order_sign(before), exempt_size_order=held_merged
    )
    return score, penalties, score - penalties, after, merges


def _held_eval(
    obs: Observation, x: float, held_r: float
) -> tuple[list[Fruit], float, float]:
    """held を x に落としたあとの (盤面, score - penalties, 本家点)。next は見ない。"""
    assert obs.held_type is not None
    before = list(obs.fruits)
    after, score, penalties, _merges, _held_merged = _evaluate_drop(
        before, obs.held_type, x, held_r, next_type=obs.next_type
    )
    return after, score - penalties, score


def _score(obs: Observation, x: float, held_r: float) -> float:
    """held を落とした盤＋ next の仮想最善手を採点する。"""
    after, value, _score = _held_eval(obs, x, held_r)
    if obs.next_type is not None:
        value += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)
    return value


def _next_eval_job(fruits: list[Fruit], next_type: int, next_r: float, nx: float) -> float:
    """next 候補 1 個ぶんの eval。プールに投げる単位。"""
    # その先の next は未知。育成免除は谷内同種だけが効く。
    _, score, penalties, _merges, _held_merged = _evaluate_drop(
        fruits, next_type, nx, next_r
    )
    return score - penalties


def _best_next_score(
    fruits: list[Fruit],
    next_type: int,
    *,
    step: float | None = None,
    pool: Executor | None = None,
) -> float:
    """next を最善列に落としたときの eval。"""
    next_r = fruit_radius(next_type)
    xs = [clamp_drop_x(nx, next_type) for nx in _candidates(fruits, next_type, next_r, step=step)]
    if not xs:
        return 0.0
    if pool is None:
        scores = [_next_eval_job(fruits, next_type, next_r, nx) for nx in xs]
    else:
        scores = list(
            pool.map(
                _next_eval_job,
                itertools.repeat(fruits),
                itertools.repeat(next_type),
                itertools.repeat(next_r),
                xs,
            )
        )
    return max(scores)


def _evaluate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    held_r: float,
    *,
    next_type: int | None = None,
) -> tuple[list[Fruit], float, float, int, bool]:
    """1 手落としたあとの盤面・本家点・減点・合成回数・held が合体したか。"""
    before = list(fruits)
    sign = _order_sign(before)
    after, merges, merge_types, held_merged, held_fruit = simulate_drop_held(
        before, drop_type, x
    )
    land_x, _land_y = landed_xy(before, after, drop_type, x, held_r, held_merged)

    score = merge_score(merge_types)
    # held (今回の手) が合体したときは、その反動で弾かれた無関係の実に対する
    # 大小順・埋め込み系の減点を掛けない。`merges >= 1` だけで見ると、held とは
    # 無関係に盤の別の場所でたまたま起きた合体まで一緒に免除してしまうので、
    # held 自身が合体に絡んだか (`held_merged`) で判定する。
    penalties = pen.board_penalties(after, sign=sign, exempt_size_order=held_merged)
    # FOREIGN_AIM は merges ではなく「真下の実が異種か」で見る。
    # 同種が真下なら合体待ちで OK。異種真上から転がって床で合体しても減点。
    penalties += pen.foreign_aim_penalty(before, x, drop_type, held_r)
    # 同点をほどくためだけの項。ここより上の項が全部並んだときに順位を決める。
    penalties += pen.center_tiebreak(x)
    if not held_merged:
        # 谷育成。合体しない手の中では、育つ見込みのある谷への着地を選ばせる。
        # 合体した手は本家点が付くので、そちらには足さない。
        penalties -= pen.valley_grow_bonus(before, land_x, drop_type, next_type)
    else:
        # 合体でできた実がどちらへ寄ったか。合体した手は大小順を免除する
        # (exempt_size_order) ので、どちら側から当てて新実をどこへ飛ばしたかを
        # 見る項が他に無い。同じ合成点なら大側へ寄せる当て方を選ばせる。
        penalties -= pen.merge_big_side_bonus(x, held_fruit, held_r, sign)
    return after, score, penalties, merges, held_merged


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
