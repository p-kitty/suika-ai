"""落下・転がり・衝突・合成 (pymunk)。

方策採点と SimEnv が共有する。
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import pymunk

from ..vision.classify import fruit_radius
from ..vision.held import DROP_HEIGHT
from ..vision.colors import MAX_FRUIT_TYPE
from ..vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from ..vision.state import Fruit

# --- 複数箇所で共有するチューニング ---
DT = 1.0 / 60.0
# 1 表示フレームあたりの物理分割。粗いと高速落下が 1px かすりを貫通する。
SUBSTEPS = 4
MAX_STEPS = int(4.0 / DT)
# 下向き正 (正規化盤面と同じ)。走査の間引き (_safe_skip) が速度の伸びしろに使う。
# 実機実測。放す高さを screenshots の実測 (DROP_START_Y) に固定すると、静止から
# 放されることから v^2 = 2g(y+h) になり、sqrt(y+h) が t の直線になる。9 本で 1418、
# ただし散らばりが撮った回で割れていて (後の 4 本 1442 / 先の 5 本 1399)、未把握の
# 系統差が 3% ある。丸めた 1400 はその中なので、精度を装わずこちらを取る。
# 高さは落下からではなく screenshots から取る (落下データは h=97 と h=115 を
# 残差 1.3% でしか区別できない)。
GRAVITY = 1400.0
# 落下開始の中心 y。実機は実の大きさによらず一定の高さから放す。screenshots 10 枚で
# 中心が -97.3 ± 3.9 に揃うのに対し、下端は -74.0 ± 9.6 とばらけるので中心基準。
# 半径に比例させると盤に入る速さが大きさでずれ、見えている区間の長さが cherry
# 1.28 倍・orange 1.09 倍と別方向に外れる。値は検出側が実測で持っているものを使う
# (二重に持つと片方だけずれる)。
DROP_START_Y = -DROP_HEIGHT
# 速度だけだと遅い creep を見逃す。settle.py と同様、静穏中の変位も見る。
SLEEP_FRAMES = 45
SLEEP_VEL = 2.0
SLEEP_ANG = 0.12
# 静穏ウィンドウ中にこれだけ動いたらやり直し (SLEEP_VEL 未満の一方向ずれ用)。
SLEEP_DRIFT = 1.0
# フルーツ同士の collision_type。壁は 0 のまま。
FRUIT_COLLISION_TYPE = 1

# --- 同種ペア走査の間引き (_MergeScan) ---
# 見積もった速度への上乗せ。接触の押し出し (space.collision_bias) は
# body.velocity に出ない微小な位置補正を生むので、そのぶん速度側に下駄を履かせる。
SCAN_SPEED_MARGIN = 60.0
# 1 度の見積もりで飛ばす上限。長く飛ばすほど、その間に起きる衝突で
# 見積もりの前提 (速度は重力でしか増えない) から離れていく。
MAX_SCAN_SKIP = 16

# 型ごとの慣性モーメント (_add_fruit)。質量 1.0・半径だけの関数なので使い回せる。
_MOMENTS: dict[int, float] = {}


@dataclass
class _BodyFruit:
    body: pymunk.Body
    shape: pymunk.Circle
    fruit_type: int
    # shape.radius は生成後に変わらない。pymunk 側の C 呼び出しを避けるためキャッシュ。
    radius: float
    # このドロップで投下した実。held 合体のひっぱ向きに使う。異種接触で消える
    # (_on_fruit_begin)。合体の反動での横ひっぱだけが目的なので、これは変えない。
    is_held_drop: bool = False
    # held に由来する系譜か。is_held_drop と違い異種接触では消えない
    # (方策の held_merged 判定専用。held が異種をかすってから同種に合体しても
    # 「held 自身の合体」として拾いたいが、盤の別の場所の無関係な合体には
    # 伝播させたくないので、合体で消えるたび新しい実へ引き継ぐだけにする)。
    is_held_lineage: bool = False


class _MergeScan:
    """同種ペアの走査を、接触しうるサブステップだけに絞る。

    `_find_merge_pair` は全実の position / velocity を pymunk のプロパティ経由で
    読むので、サブステップごとに全走査すると `simulate_drop` の時間の 58% を食う
    (実測。10 実の盤で 55,104 回呼んで、戻り値は 100% None だった)。

    走査したついでに「一番近い同種ペアの隙間」と「盤で一番速い実の速さ」を
    受け取り、どのペアも接触しえないサブステップ数だけ走査ごと飛ばす。
    飛ばすのは接触が起こりえない区間だけなので、合体のタイミングは変わらない。
    """

    __slots__ = ("_skip",)

    def __init__(self) -> None:
        self._skip = 0

    def find(self, bodies: list[_BodyFruit]) -> tuple[_BodyFruit, _BodyFruit] | None:
        if self._skip > 0:
            self._skip -= 1
            return None
        pair, gap, speed = _scan_merge_pair(bodies)
        # 見つかった手は合体して盤が変わる。次のサブステップは必ず走査し直す。
        if pair is None:
            self._skip = _safe_skip(gap, speed)
        return pair


def _safe_skip(gap: float, speed: float) -> int:
    """どのペアも接触しえないサブステップ数。

    速度は重力でしか増えない (elasticity=0 なので衝突では増えない)。2 つの実が
    近づく速さは高々 2*(v + g*t) なので、隙間 gap が閉じるまでの時間 T は
    g*T^2 + 2*v*T = gap の正の解が下限になる。そこからサブステップ数に直す。
    """
    if gap == math.inf:
        # 同種ペアがそもそも無い。合体は起こりえない。
        return MAX_SCAN_SKIP
    if gap <= 0.0:
        return 0
    v = speed + SCAN_SPEED_MARGIN
    t = (math.sqrt(v * v + GRAVITY * gap) - v) / GRAVITY
    return max(0, min(int(t / (DT / SUBSTEPS)), MAX_SCAN_SKIP))


class _QuietGate:
    """全実が遅く、かつ静穏中の位置ずれが小さいときだけ settled。"""

    __slots__ = ("frames", "anchor")

    def __init__(self) -> None:
        self.frames = 0
        # x, y を交互に並べた平坦な列。実ごとに小さい tuple を作らないため。
        self.anchor: tuple[float, ...] | None = None

    def reset(self) -> None:
        self.frames = 0
        self.anchor = None

    def update(self, bodies: list[_BodyFruit]) -> bool:
        """1 ステップ後の状態を見て、SLEEP_FRAMES 続いたら True。"""
        if not _all_quiet(bodies):
            self.reset()
            return False
        snap = _position_snapshot(bodies)
        anchor = self.anchor
        if anchor is None or _drifted(anchor, snap):
            self.anchor = snap
            self.frames = 1
        else:
            self.frames += 1
        return self.frames >= SLEEP_FRAMES


def land_y(fruits: tuple[Fruit, ...] | list[Fruit], x: float, held_r: float) -> float:
    """列 x に落としたときの中心 y (幾何)。床か、円どうしが接する位置。"""
    best = float(NORMALIZED_HEIGHT) - held_r
    for fruit in fruits:
        dx = abs(fruit.x - x)
        gap = fruit.radius + held_r
        if dx >= gap:
            continue
        dy = math.sqrt(gap * gap - dx * dx)
        best = min(best, fruit.y - dy)
    return best


def iter_simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> Iterator[tuple[list[Fruit], int, list[int]]]:
    """落下物理を 1 ステップずつ進める。view_sim アニメ用。

    盤面はクランプしない。choose_x からは呼ばない (毎ステップ export する)。
    """
    space, bodies = _build_space(fruits)
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    dropped = _add_fruit(space, bodies, fruit_type, x, DROP_START_Y)
    dropped.is_held_drop = True

    merges = 0
    merge_types: list[int] = []
    quiet = _QuietGate()
    scan = _MergeScan()
    yield _export_fruits(bodies, clamp=False), merges, list(merge_types)

    for _ in range(MAX_STEPS):
        stepped, _held_hit = _advance(space, bodies, merge_types, scan)
        merges += stepped
        if stepped:
            quiet.reset()
        yield _export_fruits(bodies, clamp=False), merges, list(merge_types)
        if quiet.update(bodies):
            break


def simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int, list[int]]:
    """列 x に落としたあとの盤面・合成回数・合成元 type 列。

    方策ホットパス。最終盤面だけ export する (アニメ用 iter は使わない)。
    """
    after, merges, merge_types, _held_merged, _held_x = simulate_drop_held(
        fruits, fruit_type, x
    )
    return after, merges, merge_types


def simulate_drop_held(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int, list[int], bool, float | None]:
    """simulate_drop に、held (今回落とした実自身) の行方を足したもの。

    足すのは (held が合体に絡んだか, held の系譜が最後に居る x)。

    held の系譜 (`is_held_lineage`) が関わる合体と、held とは無関係に盤の
    別の場所でたまたま起きた合体を区別するのに使う (`merges >= 1` だけで
    見ると両者が混ざる)。held が異種をかすってから同種に合体しても
    (`is_held_drop` は異種接触で消える) 系譜は追跡できるよう、合体で消える
    たび新しい実へ引き継ぐ。無関係な合体には伝播しない。

    x は合体後も系譜を追うので、合体で生まれた実が反動でどこまで運ばれたかが
    そのまま出る (`landed_xy` は held が消えた時点で幾何の推定に切り替わるので、
    合体の行き先はあちらからは読めない)。スイカまで育って消えたときは None。
    """
    space, bodies = _build_space(fruits)
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    dropped = _add_fruit(space, bodies, fruit_type, x, DROP_START_Y)
    dropped.is_held_drop = True
    dropped.is_held_lineage = True

    merges = 0
    merge_types: list[int] = []
    held_merged = False
    quiet = _QuietGate()
    scan = _MergeScan()

    for _ in range(MAX_STEPS):
        stepped, held_hit = _advance(space, bodies, merge_types, scan)
        merges += stepped
        held_merged = held_merged or held_hit
        if stepped:
            quiet.reset()
        if quiet.update(bodies):
            break

    return _export_fruits(bodies), merges, merge_types, held_merged, _lineage_x(bodies)


def _advance(
    space: pymunk.Space,
    bodies: list[_BodyFruit],
    merge_types: list[int],
    scan: _MergeScan | None = None,
) -> tuple[int, bool]:
    """表示 1 フレーム (= DT) 分の物理。SUBSTEPS に分割して進める。

    粗い step だと高速落下が 1px かすりを貫通して impulse 0 になる。
    戻り値はそのフレーム内の (合成回数, held が絡む合体があったか)。

    scan を渡すと、接触が起こりえないサブステップの同種ペア走査を飛ばす
    (`_MergeScan`)。渡さなければ毎サブステップ全走査する。
    """
    merges = 0
    held_merged = False
    sub_dt = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        # 接触中の同種を合成 (1 サブステップ 1 ペアまで)。
        paired = scan.find(bodies) if scan is not None else _find_merge_pair(bodies)
        if paired is not None:
            if _merge_pair(space, bodies, paired[0], paired[1], merge_types):
                held_merged = True
            merges += 1
            space.step(sub_dt)
        else:
            space.step(sub_dt)
    return merges, held_merged


def landed_xy(
    fruits_before: list[Fruit] | tuple[Fruit, ...],
    after: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
    held_r: float,
    held_merged: bool,
) -> tuple[float, float]:
    """simulate_drop 結果から、転がり後のおおよその着地 (x, y) を取る。

    held 自身が合体して消えたときだけ幾何の初期推定を返す。生き残っていれば
    盤に残っている同 type から実際の静止位置を拾う。盤の別の場所で無関係な
    合体が起きただけなら held は残っているので、そちらは実位置を返す
    (合体の有無 `merges` で切ると、この場合に捏造した推定値を返してしまい、
    着地位置を受け取る側 (`valley_grow_ok`) が
    嘘の座標で動く)。
    """
    x0 = max(held_r, min(NORMALIZED_WIDTH - held_r, x))
    est_y = land_y(fruits_before, x0, held_r)
    if not held_merged:
        cands = [f for f in after if f.type == fruit_type]
        if cands:
            best = min(cands, key=lambda f: abs(f.x - x0) + abs(f.y - est_y))
            return best.x, best.y
    return x0, est_y


def preview_land(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
    held_r: float,
) -> tuple[float, float]:
    """落下列 x の着地 (x, y)。内部で 1 回 simulate_drop する。"""
    x0 = max(held_r, min(NORMALIZED_WIDTH - held_r, x))
    after, _merges, _types, held_merged, _held_x = simulate_drop_held(
        fruits, fruit_type, x0
    )
    return landed_xy(fruits, after, fruit_type, x0, held_r, held_merged)


def _on_fruit_begin(
    arbiter: pymunk.Arbiter, _space: pymunk.Space, _data: object
) -> None:
    """同種は物理衝突オフ。異種に触れた held は特別扱いを外す。"""
    a, b = arbiter.shapes
    ta = getattr(a, "fruit_type", None)
    tb = getattr(b, "fruit_type", None)
    if ta is None or tb is None:
        return
    if ta == tb:
        # 合成ループだけが扱う (先に弾かれるのを防ぐ)。
        arbiter.process_collision = False
        return
    # 異種接触: 以降の合体は盤面どうしと同じ (横ひっぱなし)。
    for shape in (a, b):
        item = getattr(shape, "fruit_item", None)
        if item is not None and item.is_held_drop:
            item.is_held_drop = False


def _build_space(
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> tuple[pymunk.Space, list[_BodyFruit]]:
    space_damping = 1.0

    space = pymunk.Space()
    # y 下向き (正規化盤面と同じ)。
    space.gravity = (0.0, GRAVITY)
    space.damping = space_damping
    # 同種は衝突オフ、異種接触で held フラグを落とす。
    space.on_collision(
        collision_type_a=FRUIT_COLLISION_TYPE,
        collision_type_b=FRUIT_COLLISION_TYPE,
        begin=_on_fruit_begin,
    )

    static = space.static_body
    floor = pymunk.Segment(
        static, (0.0, NORMALIZED_HEIGHT), (NORMALIZED_WIDTH, NORMALIZED_HEIGHT), 2.0
    )
    left = pymunk.Segment(static, (0.0, -200.0), (0.0, NORMALIZED_HEIGHT), 2.0)
    right = pymunk.Segment(
        static,
        (NORMALIZED_WIDTH, -200.0),
        (NORMALIZED_WIDTH, NORMALIZED_HEIGHT),
        2.0,
    )
    for seg in (floor, left, right):
        seg.friction = 1.0
        seg.elasticity = 0.0
        space.add(seg)

    bodies: list[_BodyFruit] = []
    for fruit in fruits:
        _add_fruit(space, bodies, fruit.type, fruit.x, fruit.y, wake=False)
    return space, bodies


def _add_fruit(
    space: pymunk.Space,
    bodies: list[_BodyFruit],
    fruit_type: int,
    x: float,
    y: float,
    *,
    wake: bool = True,
) -> _BodyFruit:
    # 本家同様、全サイズ同じ質量。
    fruit_mass = 1.0

    r = fruit_radius(fruit_type)
    # 慣性モーメントは半径だけで決まるので型ごとに 1 度だけ計算する
    # (simulate_drop は候補ごとに盤を組み直すので、1 手で数百回呼ばれる)。
    moment = _MOMENTS.get(fruit_type)
    if moment is None:
        moment = pymunk.moment_for_circle(fruit_mass, 0.0, r)
        _MOMENTS[fruit_type] = moment
    body = pymunk.Body(fruit_mass, moment)
    body.position = (x, y)
    if not wake:
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0
    shape = pymunk.Circle(body, r)
    shape.friction = 0.5
    shape.elasticity = 0.0
    shape.collision_type = FRUIT_COLLISION_TYPE
    shape.fruit_type = fruit_type
    space.add(body, shape)
    item = _BodyFruit(body=body, shape=shape, fruit_type=fruit_type, radius=r)
    shape.fruit_item = item
    bodies.append(item)
    return item


def _remove_fruit(
    space: pymunk.Space, bodies: list[_BodyFruit], item: _BodyFruit
) -> None:
    if item.shape in space.shapes:
        space.remove(item.shape)
    if item.body in space.bodies:
        space.remove(item.body)
    bodies.remove(item)


def _held_in_merge(a: _BodyFruit, b: _BodyFruit) -> _BodyFruit | None:
    """held が絡む合体ならその held。盤面どうしは None。"""
    if a.is_held_drop != b.is_held_drop:
        return a if a.is_held_drop else b
    return None


def _merge_pair(
    space: pymunk.Space,
    bodies: list[_BodyFruit],
    a: _BodyFruit,
    b: _BodyFruit,
    merge_types: list[int],
) -> bool:
    """同種 2 個を合成。新実は両中心の中点に出す (実機と同じ)。

    戻り値は held の系譜 (`is_held_lineage`) が絡んだか。合体で消える a/b から
    新実へ引き継ぐので、held が異種をかすってから合体しても拾える
    (横ひっぱ用の `is_held_drop` は異種接触で消えるので別扱い)。
    """
    # held 合体の横ひっぱ。移動量大 (ギリギリ側面) ほど強い。
    side_min = 0.08
    travel_gain = 7.0
    speed_gain = 0.06

    source = a.fruit_type
    new_type = source + 1
    merge_types.append(source)

    ma = a.body.mass
    mb = b.body.mass
    pa = a.body.position
    pb = b.body.position
    va = a.body.velocity
    vb = b.body.velocity
    # 質量や運動エネルギーで寄せず、接触した 2 中心の幾何中点。
    mid_x = 0.5 * (pa.x + pb.x)
    mid_y = 0.5 * (pa.y + pb.y)
    parent_m = ma + mb
    # 運動量は相殺 (平均)。held 合体だけ後で横ひっぱを足す。
    px = ma * va.x + mb * vb.x
    py = ma * va.y + mb * vb.y
    ang = ma * a.body.angular_velocity + mb * b.body.angular_velocity
    held = _held_in_merge(a, b)
    held_lineage = a.is_held_lineage or b.is_held_lineage

    _remove_fruit(space, bodies, a)
    _remove_fruit(space, bodies, b)
    if new_type > MAX_FRUIT_TYPE:
        return held_lineage

    new = _add_fruit(space, bodies, new_type, mid_x, mid_y)
    new.is_held_lineage = held_lineage
    vx = px / parent_m
    vy = py / parent_m
    aw = ang / parent_m

    if held is not None:
        other = b if held is a else a
        horiz = abs(held.body.position.x - other.body.position.x)
        touch = max(held.shape.radius + other.shape.radius, 1e-6)
        side_frac = horiz / touch
        if side_frac >= side_min:
            # held 側へ。勢いの主因は中点までの移動量 (ギリギリほど大きい)。
            side = 1.0 if held.body.position.x >= other.body.position.x else -1.0
            travel = horiz * 0.5
            speed = math.hypot(held.body.velocity.x, held.body.velocity.y)
            pull = travel * travel_gain + speed * speed_gain * side_frac * side_frac
            vx += side * pull
            aw += side * pull * 0.02

    new.body.velocity = (vx, vy)
    new.body.angular_velocity = aw
    return held_lineage


def _find_merge_pair(bodies: list[_BodyFruit]) -> tuple[_BodyFruit, _BodyFruit] | None:
    """接触中の同種ペアを 1 組選ぶ。

    上側 (小さい y) を何があっても最優先。同高さなら進行方向 (vx) 側。
    """
    return _scan_merge_pair(bodies)[0]


def _scan_merge_pair(
    bodies: list[_BodyFruit],
) -> tuple[tuple[_BodyFruit, _BodyFruit] | None, float, float]:
    """`_find_merge_pair` の本体。走査のついでに間引きの材料も返す。

    戻り値は (選んだペア, 一番近い同種ペアの隙間, 盤で一番速い実の速さ)。
    後ろの 2 つは `_MergeScan` が「次に接触しうるのは何サブステップ後か」を
    見積もるのに使う。隙間は同種ペアが無ければ inf。

    pymunk の position/velocity はプロパティ経由で C を呼ぶので、ペアごとに
    毎回読み直さず、各実 1 回だけ読んでローカルにキャッシュしてから比較する。
    """
    n = len(bodies)
    types = [item.fruit_type for item in bodies]
    radii = [item.radius for item in bodies]
    xs = [0.0] * n
    ys = [0.0] * n
    vxs = [0.0] * n
    vys = [0.0] * n
    speeds = [0.0] * n
    max_speed = 0.0
    for i, item in enumerate(bodies):
        pos = item.body.position
        xs[i] = pos.x
        ys[i] = pos.y
        vel = item.body.velocity
        vxs[i] = vel.x
        vys[i] = vel.y
        speed = math.hypot(vel.x, vel.y)
        speeds[i] = speed
        if speed > max_speed:
            max_speed = speed

    best: tuple[_BodyFruit, _BodyFruit] | None = None
    best_key: tuple[float, int, float] | None = None
    # 接触していない同種ペアの最小隙間。間引きの見積もりに使う。
    min_gap = math.inf
    for i in range(n):
        ti = types[i]
        xi, yi, vxi, ri = xs[i], ys[i], vxs[i], radii[i]
        for j in range(i + 1, n):
            if ti != types[j]:
                continue
            xj, yj = xs[j], ys[j]
            touch = ri + radii[j]
            dist = math.hypot(xi - xj, yi - yj)
            if dist > touch:
                gap = dist - touch
                if gap < min_gap:
                    min_gap = gap
                continue
            # 上 (小さい y) を最優先。同高さなら動いている側の進行方向。
            if speeds[i] >= speeds[j]:
                ref_x, ref_vx, other_x = xi, vxi, xj
            else:
                ref_x, ref_vx, other_x = xj, vxs[j], xi
            dx = other_x - ref_x
            in_dir = 0 if abs(ref_vx) >= 1.0 and dx * ref_vx > 0.0 else 1
            upper_y = yi if yi <= yj else yj
            key = (upper_y, in_dir, dist / max(touch, 1e-6))
            if best_key is None or key < best_key:
                best_key = key
                best = (bodies[i], bodies[j])
    return best, min_gap, max_speed


def _position_snapshot(bodies: list[_BodyFruit]) -> tuple[float, ...]:
    """全実の位置を x, y 交互の平坦な列にする。"""
    snap: list[float] = []
    for item in bodies:
        pos = item.body.position
        snap.append(pos.x)
        snap.append(pos.y)
    return tuple(snap)


def _drifted(anchor: tuple[float, ...], current: tuple[float, ...]) -> bool:
    """静穏開始位置から SLEEP_DRIFT を超えて動いた実があるか。

    個数が変わっていたらドリフト扱い (やり直し)。
    """
    if len(anchor) != len(current):
        return True
    for i in range(0, len(anchor), 2):
        if math.hypot(anchor[i] - current[i], anchor[i + 1] - current[i + 1]) > SLEEP_DRIFT:
            return True
    return False


def _all_quiet(bodies: list[_BodyFruit]) -> bool:
    """全実が速度・角速度とも閾値未満か。

    後ろから見る。bodies は追加順で、落下中の実 (今回のドロップ) が末尾にいるので、
    落ちているあいだは 1 個目で False を返せる。全実の velocity は pymunk の
    プロパティ経由で C を呼ぶので、その読み取りを丸ごと省ける。全実の AND を
    取るだけの述語なので、見る順番で結果は変わらない。
    """
    for item in reversed(bodies):
        body = item.body
        v = body.velocity
        if math.hypot(v.x, v.y) > SLEEP_VEL:
            return False
        if abs(body.angular_velocity) > SLEEP_ANG:
            return False
    return True


def _lineage_x(bodies: list[_BodyFruit]) -> float | None:
    """held の系譜が最後に居る x。系譜が盤に残っていなければ None。

    クランプは `_export_fruits` と揃える (壁でめり込んだぶんを盤の外の座標で
    返すと、呼び元が after の同じ実と突き合わせたときにずれる)。
    """
    for item in bodies:
        if item.is_held_lineage:
            r = item.radius
            return max(r, min(NORMALIZED_WIDTH - r, float(item.body.position.x)))
    return None


def _export_fruits(bodies: list[_BodyFruit], *, clamp: bool = True) -> list[Fruit]:
    out: list[Fruit] = []
    for item in bodies:
        x = float(item.body.position.x)
        y = float(item.body.position.y)
        r = float(item.shape.radius)
        if clamp:
            # 床・壁で少しめり込むので軽くクランプ。
            x = max(r, min(NORMALIZED_WIDTH - r, x))
            y = max(r * 0.1, min(NORMALIZED_HEIGHT - r, y))
        out.append(
            Fruit(
                type=item.fruit_type,
                x=x,
                y=y,
                radius=r,
                confidence=100.0,
            )
        )
    out.sort(key=lambda f: (f.y, f.x))
    return out
