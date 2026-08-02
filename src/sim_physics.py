"""落下・転がり・衝突・合成 (pymunk)。

方策採点と SimEnv が共有する。
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import pymunk

from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# --- 複数箇所で共有するチューニング ---
DT = 1.0 / 60.0
# 1 表示フレームあたりの物理分割。粗いと高速落下が 1px かすりを貫通する。
SUBSTEPS = 4
MAX_STEPS = int(4.0 / DT)
# 速度だけだと遅い creep を見逃す。settle.py と同様、静穏中の変位も見る。
SLEEP_FRAMES = 45
SLEEP_VEL = 2.0
SLEEP_ANG = 0.12
# 静穏ウィンドウ中にこれだけ動いたらやり直し (SLEEP_VEL 未満の一方向ずれ用)。
SLEEP_DRIFT = 1.0
# フルーツ同士の collision_type。壁は 0 のまま。
FRUIT_COLLISION_TYPE = 1


@dataclass
class _BodyFruit:
    body: pymunk.Body
    shape: pymunk.Circle
    fruit_type: int
    # このドロップで投下した実。held 合体のひっぱ向きに使う。
    is_held_drop: bool = False


class _QuietGate:
    """全実が遅く、かつ静穏中の位置ずれが小さいときだけ settled。"""

    __slots__ = ("frames", "anchor")

    def __init__(self) -> None:
        self.frames = 0
        self.anchor: tuple[tuple[float, float], ...] | None = None

    def reset(self) -> None:
        self.frames = 0
        self.anchor = None

    def update(self, bodies: list[_BodyFruit]) -> bool:
        """1 ステップ後の状態を見て、SLEEP_FRAMES 続いたら True。"""
        if not _all_quiet(bodies):
            self.reset()
            return False
        snap = tuple(
            (float(item.body.position.x), float(item.body.position.y)) for item in bodies
        )
        if self.anchor is None:
            self.anchor = snap
            self.frames = 1
        elif _max_pos_drift(self.anchor, snap) > SLEEP_DRIFT:
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
    # 盤上端より少し上から落とす。
    dropped = _add_fruit(space, bodies, fruit_type, x, -r * 1.5)
    dropped.is_held_drop = True

    merges = 0
    merge_types: list[int] = []
    quiet = _QuietGate()
    yield _export_fruits(bodies, clamp=False), merges, list(merge_types)

    for _ in range(MAX_STEPS):
        stepped = _advance(space, bodies, merge_types)
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
    space, bodies = _build_space(fruits)
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    # 盤上端より少し上から落とす。
    dropped = _add_fruit(space, bodies, fruit_type, x, -r * 1.5)
    dropped.is_held_drop = True

    merges = 0
    merge_types: list[int] = []
    quiet = _QuietGate()

    for _ in range(MAX_STEPS):
        stepped = _advance(space, bodies, merge_types)
        merges += stepped
        if stepped:
            quiet.reset()
        if quiet.update(bodies):
            break

    return _export_fruits(bodies), merges, merge_types


def _advance(
    space: pymunk.Space,
    bodies: list[_BodyFruit],
    merge_types: list[int],
) -> int:
    """表示 1 フレーム (= DT) 分の物理。SUBSTEPS に分割して進める。

    粗い step だと高速落下が 1px かすりを貫通して impulse 0 になる。
    戻り値はそのフレーム内の合成回数。
    """
    merges = 0
    sub_dt = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        # 接触中の同種を合成 (1 サブステップ 1 ペアまで)。
        paired = _find_merge_pair(bodies)
        if paired is not None:
            _merge_pair(space, bodies, paired[0], paired[1], merge_types)
            merges += 1
            space.step(sub_dt)
        else:
            space.step(sub_dt)
    return merges


def landed_xy(
    fruits_before: list[Fruit] | tuple[Fruit, ...],
    after: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
    held_r: float,
    merges: int,
) -> tuple[float, float]:
    """simulate_drop 結果から、転がり後のおおよその着地 (x, y) を取る。

    合成で消える場合もあるので、幾何の初期推定に寄せつつ
    シミュレーション後に同 type が残っていればそれを使う。
    """
    x0 = max(held_r, min(NORMALIZED_WIDTH - held_r, x))
    est_y = land_y(fruits_before, x0, held_r)
    if merges == 0:
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
    after, merges, _types = simulate_drop(fruits, fruit_type, x0)
    return landed_xy(fruits, after, fruit_type, x0, held_r, merges)


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
    gravity = 2800.0
    space_damping = 1.0
    # 床摩擦は実どうしより少し高め。ノック後の氷上滑走を抑える。
    wall_friction = 0.28
    wall_elasticity = 0.08

    space = pymunk.Space()
    # y 下向き (正規化盤面と同じ)。
    space.gravity = (0.0, gravity)
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
        seg.friction = wall_friction
        seg.elasticity = wall_elasticity
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
    # 本家同様、全サイズ同じ質量。Chipmunk の摩擦は積。
    fruit_mass = 1.0
    friction = 0.22
    elasticity = 0.0

    r = fruit_radius(fruit_type)
    moment = pymunk.moment_for_circle(fruit_mass, 0.0, r)
    body = pymunk.Body(fruit_mass, moment)
    body.position = (x, y)
    if not wake:
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0
    shape = pymunk.Circle(body, r)
    shape.friction = friction
    shape.elasticity = elasticity
    shape.collision_type = FRUIT_COLLISION_TYPE
    shape.fruit_type = fruit_type
    space.add(body, shape)
    item = _BodyFruit(body=body, shape=shape, fruit_type=fruit_type)
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
) -> None:
    """同種 2 個を合成。新実は両中心の中点に出す (実機と同じ)。"""
    # held 合体の横ひっぱ。移動量大 (ギリギリ側面) ほど強い。
    side_min = 0.08
    travel_gain = 14.0
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

    _remove_fruit(space, bodies, a)
    _remove_fruit(space, bodies, b)
    if new_type > MAX_FRUIT_TYPE:
        return

    new = _add_fruit(space, bodies, new_type, mid_x, mid_y)
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


def _find_merge_pair(bodies: list[_BodyFruit]) -> tuple[_BodyFruit, _BodyFruit] | None:
    """接触中の同種ペアを 1 組選ぶ。

    上側 (小さい y) を何があっても最優先。同高さなら進行方向 (vx) 側。
    """
    best: tuple[_BodyFruit, _BodyFruit] | None = None
    best_key: tuple[float, int, float] | None = None
    n = len(bodies)
    for i in range(n):
        a = bodies[i]
        for j in range(i + 1, n):
            b = bodies[j]
            if a.fruit_type != b.fruit_type:
                continue
            ra = a.shape.radius
            rb = b.shape.radius
            touch = ra + rb
            dist = math.hypot(
                a.body.position.x - b.body.position.x,
                a.body.position.y - b.body.position.y,
            )
            if dist > touch:
                continue
            # 上 (小さい y) を最優先。同高さなら動いている側の進行方向。
            sa = math.hypot(a.body.velocity.x, a.body.velocity.y)
            sb = math.hypot(b.body.velocity.x, b.body.velocity.y)
            ref, other = (a, b) if sa >= sb else (b, a)
            vx = ref.body.velocity.x
            dx = other.body.position.x - ref.body.position.x
            in_dir = 0 if abs(vx) >= 1.0 and dx * vx > 0.0 else 1
            upper_y = min(a.body.position.y, b.body.position.y)
            key = (upper_y, in_dir, dist / max(touch, 1e-6))
            if best_key is None or key < best_key:
                best_key = key
                best = (a, b)
    return best


def _max_pos_drift(
    anchor: tuple[tuple[float, float], ...],
    current: tuple[tuple[float, float], ...],
) -> float:
    """静穏開始位置からの最大変位。個数が変わったら無限大 (やり直し)。"""
    if len(anchor) != len(current):
        return math.inf
    best = 0.0
    for (ax, ay), (bx, by) in zip(anchor, current):
        best = max(best, math.hypot(ax - bx, ay - by))
    return best


def _all_quiet(bodies: list[_BodyFruit]) -> bool:
    if not bodies:
        return True
    for item in bodies:
        v = item.body.velocity
        speed = math.hypot(v.x, v.y)
        if speed > SLEEP_VEL:
            return False
        if abs(item.body.angular_velocity) > SLEEP_ANG:
            return False
    return True


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
