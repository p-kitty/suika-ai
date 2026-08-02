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

# --- Tuning shared across several places ---
DT = 1.0 / 60.0
# Physics subdivision per displayed frame. Too coarse and a fast fall passes through a 1px graze.
SUBSTEPS = 4
MAX_STEPS = int(4.0 / DT)
# Speed alone misses slow creep. Like settle.py, displacement during quiet is checked too.
SLEEP_FRAMES = 45
SLEEP_VEL = 2.0
SLEEP_ANG = 0.12
# Restart when it moves this much during the quiet window (for one-directional drift below SLEEP_VEL).
SLEEP_DRIFT = 1.0
# collision_type between fruits. Walls stay 0.
FRUIT_COLLISION_TYPE = 1


@dataclass
class _BodyFruit:
    body: pymunk.Body
    shape: pymunk.Circle
    fruit_type: int
    # The fruit dropped this drop. Used for the direction of the held merge pull.
    is_held_drop: bool = False


class _QuietGate:
    """Settled only when every fruit is slow and the position drift during quiet is small."""

    __slots__ = ("frames", "anchor")

    def __init__(self) -> None:
        self.frames = 0
        self.anchor: tuple[tuple[float, float], ...] | None = None

    def reset(self) -> None:
        self.frames = 0
        self.anchor = None

    def update(self, bodies: list[_BodyFruit]) -> bool:
        """Look at the state after one step, and return True once it has lasted SLEEP_FRAMES."""
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
    """Physics for one displayed frame (= DT). Advanced in SUBSTEPS pieces.

    A coarse step lets a fast fall pass through a 1px graze with impulse 0.
    Returns the merge count within that frame.
    """
    merges = 0
    sub_dt = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        # Merge touching same types (at most 1 pair per substep).
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


def _on_fruit_begin(
    arbiter: pymunk.Arbiter, _space: pymunk.Space, _data: object
) -> None:
    """Same types have physical collision off. held touching a different type loses its special treatment."""
    a, b = arbiter.shapes
    ta = getattr(a, "fruit_type", None)
    tb = getattr(b, "fruit_type", None)
    if ta is None or tb is None:
        return
    if ta == tb:
        # Only the merge loop handles it (prevents being knocked away first).
        arbiter.process_collision = False
        return
    # Contact with a different type: later merges are the same as board-to-board (no sideways pull).
    for shape in (a, b):
        item = getattr(shape, "fruit_item", None)
        if item is not None and item.is_held_drop:
            item.is_held_drop = False


def _build_space(
    fruits: list[Fruit] | tuple[Fruit, ...],
) -> tuple[pymunk.Space, list[_BodyFruit]]:
    gravity = 2800.0
    space_damping = 1.0
    # Floor friction slightly higher than between fruits. Suppresses sliding on ice after a knock.
    wall_friction = 0.28
    wall_elasticity = 0.08

    space = pymunk.Space()
    # y points down (same as the normalized board).
    space.gravity = (0.0, gravity)
    space.damping = space_damping
    # Same types collide off; contact with a different type drops the held flag.
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
    # Like the real game, every size has the same mass. Chipmunk friction is the product.
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
    # Sideways pull of held merges. Stronger the larger the movement (a narrow side graze).
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
        if side_frac >= side_min:
            # Toward held. The main source of momentum is the movement to the midpoint (larger for narrow grazes).
            side = 1.0 if held.body.position.x >= other.body.position.x else -1.0
            travel = horiz * 0.5
            speed = math.hypot(held.body.velocity.x, held.body.velocity.y)
            pull = travel * travel_gain + speed * speed_gain * side_frac * side_frac
            vx += side * pull
            aw += side * pull * 0.02

    new.body.velocity = (vx, vy)
    new.body.angular_velocity = aw


def _find_merge_pair(bodies: list[_BodyFruit]) -> tuple[_BodyFruit, _BodyFruit] | None:
    """Pick one touching same-type pair.

    The upper one (smaller y) takes top priority no matter what. At the same height, the side of travel (vx).
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
            # The top (smaller y) takes priority. At the same height, the direction of travel of the moving side.
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
    """The max displacement from the position at the start of quiet. Infinity if the count changed (start over)."""
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
