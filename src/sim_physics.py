"""Falling, rolling, collision and merging (pymunk).

Shared by policy scoring and SimEnv.
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

# --- Tuning (leaning toward the real game's rolling; pymunk uses the product of frictions) ---
GRAVITY = 2800.0
DT = 1.0 / 60.0
# Max simulated time per drop.
MAX_SIM_SECONDS = 4.0
MAX_STEPS = int(MAX_SIM_SECONDS / DT)
# How many consecutive quiet frames count as still.
SLEEP_FRAMES = 22
VEL_SLEEP = 8.0
ANG_SLEEP = 0.45
# Chipmunk multiplies the friction of both shapes (effective ≈ product).
FRICTION = 0.08
ELASTICITY = 0.20
# Walls and floor
WALL_FRICTION = 0.10
WALL_ELASTICITY = 0.08
# Space damping (1 = none).
SPACE_DAMPING = 1.0
# Overlap margin for the merge check (ratio to the sum of radii).
MERGE_SLOP = 1.02
# Held merge: below this sideways offset ratio it counts as directly above (no sideways pull).
MERGE_SIDE_MIN = 0.08
# Held merge pull: velocity per px of sideways movement to the midpoint.
# Weak for small offsets, strong for narrow side grazes (large movement).
MERGE_TRAVEL_GAIN = 14.0
# Speed helper (small). Adds a little only at narrow sides with offset^2.
MERGE_SPEED_GAIN = 0.06
# collision_type between fruits. Walls stay 0.
FRUIT_COLLISION_TYPE = 1
# Mass = density * area. Bigger is harder to push.
DENSITY = 0.07


@dataclass
class _BodyFruit:
    body: pymunk.Body
    shape: pymunk.Circle
    fruit_type: int
    # The fruit dropped this drop. Used for the direction of the held merge pull.
    is_held_drop: bool = False


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


def iter_simulate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> Iterator[tuple[list[Fruit], int, list[int]]]:
    """Advance the fall physics one step at a time. For view_sim animation.

    The board is not clamped. Not called from choose_x (it exports every step).
    """
    space, bodies = _build_space(fruits)
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    # Drop from slightly above the top of the board.
    dropped = _add_fruit(space, bodies, fruit_type, x, -r * 1.5)
    dropped.is_held_drop = True

    merges = 0
    merge_types: list[int] = []
    quiet = 0
    yield _export_fruits(bodies, clamp=False), merges, list(merge_types)

    for _ in range(MAX_STEPS):
        # Merge touching same types (at most 1 pair per step).
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
    """Board after dropping at column x, merge count and the list of source types merged.

    The policy hot path. Only exports the final board (does not use the animation iter).
    """
    space, bodies = _build_space(fruits)
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    # Drop from slightly above the top of the board.
    dropped = _add_fruit(space, bodies, fruit_type, x, -r * 1.5)
    dropped.is_held_drop = True

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

    return _export_fruits(bodies), merges, merge_types


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


def _ignore_same_type(
    arbiter: pymunk.Arbiter, _space: pymunk.Space, _data: object
) -> None:
    """Same types do not collide physically; only the merge loop handles them (prevents being knocked away first)."""
    a, b = arbiter.shapes
    ta = getattr(a, "fruit_type", None)
    tb = getattr(b, "fruit_type", None)
    if ta is not None and ta == tb:
        arbiter.process_collision = False


def _build_space(
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> tuple[pymunk.Space, list[_BodyFruit]]:
    space = pymunk.Space()
    # y points down (same as the normalized board).
    space.gravity = (0.0, GRAVITY)
    space.damping = SPACE_DAMPING
    # Disable collision response between same-type fruits (pymunk 7: process_collision).
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
    """That held if the merge involves held. None for board-to-board."""
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
    """Merge two of the same type. The new fruit appears at the midpoint of the two centers (same as the real game)."""
    source = a.fruit_type
    new_type = source + 1
    merge_types.append(source)

    ma = a.body.mass
    mb = b.body.mass
    pa = a.body.position
    pb = b.body.position
    va = a.body.velocity
    vb = b.body.velocity
    # Not weighted by mass or kinetic energy; the geometric midpoint of the two touching centers.
    mid_x = 0.5 * (pa.x + pb.x)
    mid_y = 0.5 * (pa.y + pb.y)
    parent_m = ma + mb
    # Momentum cancels (averaged). Only held merges add the sideways pull afterwards.
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
            # Toward held. The main source of momentum is the movement to the midpoint (larger for narrow grazes).
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


def _export_fruits(bodies: list[_BodyFruit], *, clamp: bool = True) -> list[Fruit]:
    out: list[Fruit] = []
    for item in bodies:
        x = float(item.body.position.x)
        y = float(item.body.position.y)
        r = float(item.shape.radius)
        if clamp:
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


def _export_fruits_clamped(fruits: list[Fruit]) -> list[Fruit]:
    """Clamp animation snapshots within the board."""
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
