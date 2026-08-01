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

# --- チューニング (実機の転がり寄り。pymunk は摩擦を積で使う) ---
GRAVITY = 2800.0
DT = 1.0 / 60.0
# 1 ドロップあたりの最大シミュレーション時間。
MAX_SIM_SECONDS = 4.0
MAX_STEPS = int(MAX_SIM_SECONDS / DT)
# 連続何フレーム静かなら静止とみなす。
SLEEP_FRAMES = 22
VEL_SLEEP = 8.0
ANG_SLEEP = 0.45
# Chipmunk は両 shape の friction を掛け算する (実効 ≈ 積)。
FRICTION = 0.08
ELASTICITY = 0.20
# 壁・床
WALL_FRICTION = 0.10
WALL_ELASTICITY = 0.08
# 空間減衰 (1=なし)。
SPACE_DAMPING = 1.0
# 合成判定の重なり余裕 (半径和に対する比率)。
MERGE_SLOP = 1.02
# held 合体: 横ずれ比がこれ未満なら真上扱い (横ひっぱなし)。
MERGE_SIDE_MIN = 0.08
# held 合体のひっぱり: 中点までの横移動量 (px) あたりの速度。
# わずかなずれは弱く、ギリギリ側面 (移動量大) ほど強い。
MERGE_TRAVEL_GAIN = 14.0
# 速さの補助 (小さめ)。ずれ^2 でギリギリ側だけ少し足す。
MERGE_SPEED_GAIN = 0.06
# フルーツ同士の collision_type。壁は 0 のまま。
FRUIT_COLLISION_TYPE = 1
# 本家同様、全サイズ同じ質量 (大きさで重くしない)。
FRUIT_MASS = 1.0


@dataclass
class _BodyFruit:
    body: pymunk.Body
    shape: pymunk.Circle
    fruit_type: int
    # このドロップで投下した実。held 合体のひっぱ向きに使う。
    is_held_drop: bool = False


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
    quiet = 0
    yield _export_fruits(bodies, clamp=False), merges, list(merge_types)

    for _ in range(MAX_STEPS):
        # 接触中の同種を合成 (1 ステップ 1 ペアまで)。
        paired = _find_merge_pair(bodies)
        if paired is not None:
            _merge_pair(space, bodies, paired[0], paired[1], merge_types)
            merges += 1
            quiet = 0
            space.step(DT)
        else:
            space.step(DT)
            if _all_quiet(bodies):
                quiet += 1
            else:
                quiet = 0

        yield _export_fruits(bodies, clamp=False), merges, list(merge_types)
        if quiet >= SLEEP_FRAMES:
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
    quiet = 0

    for _ in range(MAX_STEPS):
        # 接触中の同種を合成 (1 ステップ 1 ペアまで)。
        paired = _find_merge_pair(bodies)
        if paired is not None:
            _merge_pair(space, bodies, paired[0], paired[1], merge_types)
            merges += 1
            quiet = 0
            space.step(DT)
            continue

        space.step(DT)
        if _all_quiet(bodies):
            quiet += 1
            if quiet >= SLEEP_FRAMES:
                break
        else:
            quiet = 0

    return _export_fruits(bodies), merges, merge_types


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


def _ignore_same_type(
    arbiter: pymunk.Arbiter, _space: pymunk.Space, _data: object
) -> None:
    """同種は物理衝突させず、合成ループだけが扱う (先に弾かれるのを防ぐ)。"""
    a, b = arbiter.shapes
    ta = getattr(a, "fruit_type", None)
    tb = getattr(b, "fruit_type", None)
    if ta is not None and ta == tb:
        arbiter.process_collision = False


def _build_space(
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> tuple[pymunk.Space, list[_BodyFruit]]:
    space = pymunk.Space()
    # y 下向き (正規化盤面と同じ)。
    space.gravity = (0.0, GRAVITY)
    space.damping = SPACE_DAMPING
    # 同種フルーツ同士の衝突応答を無効化 (pymunk 7: process_collision)。
    space.on_collision(
        collision_type_a=FRUIT_COLLISION_TYPE,
        collision_type_b=FRUIT_COLLISION_TYPE,
        begin=_ignore_same_type,
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
        seg.friction = WALL_FRICTION
        seg.elasticity = WALL_ELASTICITY
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
    r = fruit_radius(fruit_type)
    moment = pymunk.moment_for_circle(FRUIT_MASS, 0.0, r)
    body = pymunk.Body(FRUIT_MASS, moment)
    body.position = (x, y)
    if not wake:
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0
    shape = pymunk.Circle(body, r)
    shape.friction = FRICTION
    shape.elasticity = ELASTICITY
    shape.collision_type = FRUIT_COLLISION_TYPE
    shape.fruit_type = fruit_type
    space.add(body, shape)
    item = _BodyFruit(body=body, shape=shape, fruit_type=fruit_type)
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
        if side_frac >= MERGE_SIDE_MIN:
            # held 側へ。勢いの主因は中点までの移動量 (ギリギリほど大きい)。
            side = 1.0 if held.body.position.x >= other.body.position.x else -1.0
            travel = horiz * 0.5
            speed = math.hypot(held.body.velocity.x, held.body.velocity.y)
            pull = travel * MERGE_TRAVEL_GAIN + speed * MERGE_SPEED_GAIN * side_frac * side_frac
            vx += side * pull
            aw += side * pull * 0.02

    new.body.velocity = (vx, vy)
    new.body.angular_velocity = aw


def _find_merge_pair(bodies: list[_BodyFruit]) -> tuple[_BodyFruit, _BodyFruit] | None:
    n = len(bodies)
    for i in range(n):
        a = bodies[i]
        for j in range(i + 1, n):
            b = bodies[j]
            if a.fruit_type != b.fruit_type:
                continue
            ra = a.shape.radius
            rb = b.shape.radius
            dist = math.hypot(
                a.body.position.x - b.body.position.x,
                a.body.position.y - b.body.position.y,
            )
            # スイカ同士も合成する (結果は出さず消える)。
            if dist <= (ra + rb) * MERGE_SLOP:
                return a, b
    return None


def _all_quiet(bodies: list[_BodyFruit]) -> bool:
    if not bodies:
        return True
    for item in bodies:
        v = item.body.velocity
        speed = math.hypot(v.x, v.y)
        if speed > VEL_SLEEP:
            return False
        if abs(item.body.angular_velocity) > ANG_SLEEP:
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


def _export_fruits_clamped(fruits: list[Fruit]) -> list[Fruit]:
    """アニメ用スナップショットを盤内に収める。"""
    out: list[Fruit] = []
    for fruit in fruits:
        r = fruit.radius
        out.append(
            Fruit(
                type=fruit.type,
                x=max(r, min(NORMALIZED_WIDTH - r, fruit.x)),
                y=max(r * 0.1, min(NORMALIZED_HEIGHT - r, fruit.y)),
                radius=r,
                confidence=fruit.confidence,
            )
        )
    out.sort(key=lambda f: (f.y, f.x))
    return out
