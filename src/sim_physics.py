"""落下・転がり・衝突・合成 (pymunk)。

方策採点と SimEnv が共有する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pymunk

from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# --- チューニング (見た目寄り。精度より安定・速度) ---
GRAVITY = 2800.0
DT = 1.0 / 60.0
# 1 ドロップあたりの最大シミュレーション時間。
MAX_SIM_SECONDS = 4.0
MAX_STEPS = int(MAX_SIM_SECONDS / DT)
# 連続何フレーム静かなら静止とみなす。
SLEEP_FRAMES = 18
VEL_SLEEP = 12.0
ANG_SLEEP = 0.8
FRICTION = 0.35
ELASTICITY = 0.12
# 壁・床
WALL_FRICTION = 0.4
WALL_ELASTICITY = 0.05
# 合成判定の重なり余裕 (半径和に対する比率)。
MERGE_SLOP = 1.02
# 質量 = 密度 * 面積。大きいほど押しにくい。
DENSITY = 0.08


@dataclass
class _BodyFruit:
    body: pymunk.Body
    shape: pymunk.Circle
    fruit_type: int


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


def simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int, list[int]]:
    """列 x に落としたあとの盤面・合成回数・合成元 type 列。"""
    space, bodies = _build_space(fruits)
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    # 盤上端より少し上から落とす。
    _add_fruit(space, bodies, fruit_type, x, -r * 1.5)

    merges = 0
    merge_types: list[int] = []
    quiet = 0

    for _ in range(MAX_STEPS):
        # 接触中の同種を合成 (1 ステップ 1 ペアまで)。
        paired = _find_merge_pair(bodies)
        if paired is not None:
            a, b = paired
            source = a.fruit_type
            new_type = source + 1
            mid_x = 0.5 * (a.body.position.x + b.body.position.x)
            mid_y = 0.5 * (a.body.position.y + b.body.position.y)
            _remove_fruit(space, bodies, a)
            _remove_fruit(space, bodies, b)
            merge_types.append(source)
            merges += 1
            if new_type <= MAX_FRUIT_TYPE:
                _add_fruit(space, bodies, new_type, mid_x, mid_y)
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

    after = _export_fruits(bodies)
    return after, merges, merge_types


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


def _build_space(
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> tuple[pymunk.Space, list[_BodyFruit]]:
    space = pymunk.Space()
    # y 下向き (正規化盤面と同じ)。
    space.gravity = (0.0, GRAVITY)
    space.damping = 0.98

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
    mass = max(0.2, DENSITY * math.pi * r * r)
    moment = pymunk.moment_for_circle(mass, 0.0, r)
    body = pymunk.Body(mass, moment)
    body.position = (x, y)
    if not wake:
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0
    shape = pymunk.Circle(body, r)
    shape.friction = FRICTION
    shape.elasticity = ELASTICITY
    shape.collision_type = 1
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


def _export_fruits(bodies: list[_BodyFruit]) -> list[Fruit]:
    out: list[Fruit] = []
    for item in bodies:
        x = float(item.body.position.x)
        y = float(item.body.position.y)
        r = float(item.shape.radius)
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
