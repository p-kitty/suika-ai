"""Falling, rolling, collision and merging (pymunk).

Shared by policy scoring and SimEnv.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pymunk

from .vision.classify import fruit_radius
from .vision.colors import MAX_FRUIT_TYPE
from .vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from .vision.state import Fruit

# --- Tuning (leaning toward looks; stability and speed over precision) ---
GRAVITY = 2800.0
DT = 1.0 / 60.0
# Max simulated time per drop.
MAX_SIM_SECONDS = 4.0
MAX_STEPS = int(MAX_SIM_SECONDS / DT)
# How many consecutive quiet frames count as still.
SLEEP_FRAMES = 18
VEL_SLEEP = 12.0
ANG_SLEEP = 0.8
FRICTION = 0.35
ELASTICITY = 0.12
# Walls and floor
WALL_FRICTION = 0.4
WALL_ELASTICITY = 0.05
# Overlap margin for the merge check (ratio to the sum of radii).
MERGE_SLOP = 1.02
# Shift the merge position toward the one with more kinetic energy (0 = center of mass, 1 = up to that fruit's position).
MERGE_PULL_BIAS = 0.35
# Shift only when the sideways offset is at least this multiple × the smaller radius (avoids cascades on straight-down drops).
MERGE_PULL_MIN_HORIZ = 0.7
# Velocity multiplier inherited in merges where only one side is moving.
MERGE_VEL_SCALE = 0.55
# Mass = density * area. Bigger is harder to push.
DENSITY = 0.08


@dataclass
class _BodyFruit:
    body: pymunk.Body
    shape: pymunk.Circle
    fruit_type: int


def land_y(fruits: tuple[Fruit, ...] | list[Fruit], x: float, held_r: float) -> float:
    """Center y when dropped at column x (geometry). The floor, or where the circles touch."""
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
    """Board after dropping at column x, merge count and the list of source types merged."""
    space, bodies = _build_space(fruits)
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    # Drop from slightly above the top of the board.
    _add_fruit(space, bodies, fruit_type, x, -r * 1.5)

    merges = 0
    merge_types: list[int] = []
    quiet = 0

    for _ in range(MAX_STEPS):
        # Merge touching same types (at most 1 pair per step).
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
    """Get the approximate landing (x, y) after rolling from a simulate_drop result.

    It may disappear in a merge, so while leaning on the initial geometric estimate,
    a same type remaining after the simulation is used if present.
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
    """Landing (x, y) for drop column x. Runs simulate_drop once internally."""
    x0 = max(held_r, min(NORMALIZED_WIDTH - held_r, x))
    after, merges, _types = simulate_drop(fruits, fruit_type, x0)
    return landed_xy(fruits, after, fruit_type, x0, held_r, merges)


def _build_space(
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> tuple[pymunk.Space, list[_BodyFruit]]:
    space = pymunk.Space()
    # y points down (same as the normalized board).
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


def _merge_pair(
    space: pymunk.Space,
    bodies: list[_BodyFruit],
    a: _BodyFruit,
    b: _BodyFruit,
    merge_types: list[int],
) -> None:
    """Merge two of the same type. Reproduces the pull after merging with momentum and a slight position shift."""
    source = a.fruit_type
    new_type = source + 1
    merge_types.append(source)

    ma = a.body.mass
    mb = b.body.mass
    total_m = ma + mb
    pa = a.body.position
    pb = b.body.position
    va = a.body.velocity
    vb = b.body.velocity

    base_x = (ma * pa.x + mb * pb.x) / total_m
    base_y = (ma * pa.y + mb * pb.y) / total_m
    ke_a = ma * (va.x * va.x + va.y * va.y)
    ke_b = mb * (vb.x * vb.x + vb.y * vb.y)
    ke_sum = ke_a + ke_b
    ra = a.shape.radius
    rb = b.shape.radius
    horiz = abs(pa.x - pb.x)
    if ke_sum > 1e-6 and horiz > min(ra, rb) * MERGE_PULL_MIN_HORIZ:
        bias = ke_b / ke_sum
        pull_x = pa.x * (1.0 - bias) + pb.x * bias
        pull_y = pa.y * (1.0 - bias) + pb.y * bias
        mid_x = base_x + MERGE_PULL_BIAS * (pull_x - base_x)
        mid_y = base_y + MERGE_PULL_BIAS * (pull_y - base_y)
    else:
        mid_x = base_x
        mid_y = base_y

    px = ma * va.x + mb * vb.x
    py = ma * va.y + mb * vb.y
    ang = ma * a.body.angular_velocity + mb * b.body.angular_velocity

    _remove_fruit(space, bodies, a)
    _remove_fruit(space, bodies, b)
    if new_type > MAX_FRUIT_TYPE:
        return

    new = _add_fruit(space, bodies, new_type, mid_x, mid_y)
    new_m = new.body.mass
    if (
        ke_sum > 1e-6
        and horiz > min(ra, rb) * MERGE_PULL_MIN_HORIZ
    ):
        ke_bias = abs(ke_b - ke_a) / ke_sum
        scale = MERGE_VEL_SCALE * ke_bias
        # Initial vertical velocity tends to bounce into neighbors, so inherit only horizontal.
        new.body.velocity = (px / new_m * scale, 0.0)
        new.body.angular_velocity = ang / new_m * scale


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
            # Watermelons merge with each other too (no result, they disappear).
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
        # It sinks slightly into the floor and walls, so clamp lightly.
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
