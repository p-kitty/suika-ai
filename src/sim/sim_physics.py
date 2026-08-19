"""Falling, rolling, collision and merging (pymunk).

Shared by policy scoring and SimEnv.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import pymunk

from ..vision.classify import fruit_radius
from ..vision.colors import MAX_FRUIT_TYPE
from ..vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from ..vision.state import Fruit

# --- Tuning shared across several places ---
DT = 1.0 / 60.0
# Physics subdivision per displayed frame. Too coarse and a fast fall passes through a 1px graze.
SUBSTEPS = 4
MAX_STEPS = int(4.0 / DT)
# Positive downward (same as the normalized board). The scan skipping (_safe_skip) uses it for the headroom in speed.
GRAVITY = 2800.0
# Speed alone misses slow creep. Like settle.py, displacement during quiet is checked too.
SLEEP_FRAMES = 45
SLEEP_VEL = 2.0
SLEEP_ANG = 0.12
# Restart when it moves this much during the quiet window (for one-directional drift below SLEEP_VEL).
SLEEP_DRIFT = 1.0
# collision_type between fruits. Walls stay 0.
FRUIT_COLLISION_TYPE = 1

# --- Skipping same-type pair scans (_MergeScan) ---
# Added to the estimated speed. Contact push-out (space.collision_bias) produces small position corrections
# that do not show up in body.velocity, so the speed side gets a margin for that.
SCAN_SPEED_MARGIN = 60.0
# Upper limit skipped per estimate. The longer the skip, the more collisions during it
# drift from the estimate's premise (speed only increases through gravity).
MAX_SCAN_SKIP = 16

# Moment of inertia per type (_add_fruit). With mass 1.0 it is a function of radius only, so it can be reused.
_MOMENTS: dict[int, float] = {}


@dataclass
class _BodyFruit:
    body: pymunk.Body
    shape: pymunk.Circle
    fruit_type: int
    # shape.radius does not change after creation. Cached to avoid calls into C on the pymunk side.
    radius: float
    # The fruit dropped this drop. Used for the sideways pull of held merges. Cleared on contact with a different type
    # (_on_fruit_begin). Its only purpose is the sideways pull from merge recoil, so this is not changed.
    is_held_drop: bool = False
    # Whether it descends from held. Unlike is_held_drop it is not cleared on contact with a different type
    # (only for the policy's held_merged check. Even if held grazes a different type and then merges with the same type,
    # we want to catch it as 'held's own merge', but not propagate it to unrelated merges elsewhere on the board,
    # so it is only handed over to the new fruit each time it disappears in a merge).
    is_held_lineage: bool = False


class _MergeScan:
    """Limit same-type pair scans to the substeps where contact is possible.

    `_find_merge_pair` reads the position / velocity of every fruit through pymunk properties,
    so scanning everything every substep eats 58% of the time of `simulate_drop`
    (measured. On a 10-fruit board it was called 55,104 times and returned None 100% of the time).

    While scanning, it receives 'the gap of the nearest same-type pair' and 'the speed of the fastest fruit on the board',
    and skips the scan for as many substeps as no pair can possibly touch.
    Only stretches where contact cannot happen are skipped, so merge timing does not change.
    """

    __slots__ = ("_skip",)

    def __init__(self) -> None:
        self._skip = 0

    def find(self, bodies: list[_BodyFruit]) -> tuple[_BodyFruit, _BodyFruit] | None:
        if self._skip > 0:
            self._skip -= 1
            return None
        pair, gap, speed = _scan_merge_pair(bodies)
        # When found, the merge changes the board. Always rescan on the next substep.
        if pair is None:
            self._skip = _safe_skip(gap, speed)
        return pair


def _safe_skip(gap: float, speed: float) -> int:
    """The number of substeps in which no pair can touch.

    Speed only increases through gravity (elasticity=0, so collisions do not add speed). The speed at which two fruits
    approach is at most 2*(v + g*t), so the time T for the gap to close is bounded below by
    the positive root of g*T^2 + 2*v*T = gap. That is converted to a number of substeps.
    """
    if gap == math.inf:
        # No same-type pair at all. A merge cannot happen.
        return MAX_SCAN_SKIP
    if gap <= 0.0:
        return 0
    v = speed + SCAN_SPEED_MARGIN
    t = (math.sqrt(v * v + GRAVITY * gap) - v) / GRAVITY
    return max(0, min(int(t / (DT / SUBSTEPS)), MAX_SCAN_SKIP))


class _QuietGate:
    """Settled only when every fruit is slow and the position drift during quiet is small."""

    __slots__ = ("frames", "anchor")

    def __init__(self) -> None:
        self.frames = 0
        # A flat list alternating x, y. Avoids creating a small tuple per fruit.
        self.anchor: tuple[float, ...] | None = None

    def reset(self) -> None:
        self.frames = 0
        self.anchor = None

    def update(self, bodies: list[_BodyFruit]) -> bool:
        """Look at the state after one step, and return True once it has lasted SLEEP_FRAMES."""
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
    """Board after dropping at column x, merge count and the list of source types merged.

    The policy hot path. Only exports the final board (does not use the animation iter).
    """
    after, merges, merge_types, _held_merged = simulate_drop_held(fruits, fruit_type, x)
    return after, merges, merge_types


def simulate_drop_held(
    fruits: list[Fruit] | tuple[Fruit, ...],
    fruit_type: int,
    x: float,
) -> tuple[list[Fruit], int, list[int], bool]:
    """simulate_drop plus whether held (the fruit dropped this time) took part in a merge.

    Used to tell merges involving held's lineage (`is_held_lineage`) from merges that happened by chance
    elsewhere on the board unrelated to held (looking only at `merges >= 1`
    mixes the two). Even if held grazes a different type and then merges with the same type
    (`is_held_drop` is cleared on contact with a different type), the lineage can be tracked because it is handed over to the new fruit
    each time it disappears in a merge. It does not propagate to unrelated merges.
    """
    space, bodies = _build_space(fruits)
    r = fruit_radius(fruit_type)
    x = max(r, min(NORMALIZED_WIDTH - r, x))
    # Drop from slightly above the top of the board.
    dropped = _add_fruit(space, bodies, fruit_type, x, -r * 1.5)
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

    return _export_fruits(bodies), merges, merge_types, held_merged


def _advance(
    space: pymunk.Space,
    bodies: list[_BodyFruit],
    merge_types: list[int],
    scan: _MergeScan | None = None,
) -> tuple[int, bool]:
    """Physics for one displayed frame (= DT). Advanced in SUBSTEPS pieces.

    A coarse step lets a fast fall pass through a 1px graze with impulse 0.
    Returns (merge count, whether a merge involving held occurred) within that frame.

    Passing scan skips same-type pair scans in substeps where contact cannot happen
    (`_MergeScan`). Without it every substep scans everything.
    """
    merges = 0
    held_merged = False
    sub_dt = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        # Merge touching same types (at most 1 pair per substep).
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
    """Get the approximate landing (x, y) after rolling from a simulate_drop result.

    Returns the initial geometric estimate only when held itself merged and disappeared. If it survived,
    its actual resting position is picked from the same type remaining on the board. If only an unrelated merge
    happened elsewhere on the board, held remains, so its actual position is returned
    (cutting on the merge count `merges` would return a fabricated estimate in this case,
    the side receiving the landing position (`valley_grow_ok`) would
    act on false coordinates).
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
    """Landing (x, y) for drop column x. Runs simulate_drop once internally."""
    x0 = max(held_r, min(NORMALIZED_WIDTH - held_r, x))
    after, _merges, _types, held_merged = simulate_drop_held(fruits, fruit_type, x0)
    return landed_xy(fruits, after, fruit_type, x0, held_r, held_merged)


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
    space_damping = 1.0

    space = pymunk.Space()
    # y points down (same as the normalized board).
    space.gravity = (0.0, GRAVITY)
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
    # Like the real game, every size has the same mass.
    fruit_mass = 1.0

    r = fruit_radius(fruit_type)
    # The moment of inertia depends only on radius, so compute it once per type
    # (simulate_drop rebuilds the board per candidate, so it is called hundreds of times per move).
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
) -> bool:
    """Merge two of the same type. The new fruit appears at the midpoint of the two centers (same as the real game).

    Returns whether held's lineage (`is_held_lineage`) was involved. It is handed over from a/b, which disappear in the merge,
    to the new fruit, so it is caught even if held grazes a different type before merging
    (`is_held_drop` for the sideways pull is cleared on contact with a different type, so it is handled separately).
    """
    # Sideways pull of held merges. Stronger the larger the movement (a narrow side graze).
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
    # Not weighted by mass or kinetic energy; the geometric midpoint of the two touching centers.
    mid_x = 0.5 * (pa.x + pb.x)
    mid_y = 0.5 * (pa.y + pb.y)
    parent_m = ma + mb
    # Momentum cancels (averaged). Only held merges add the sideways pull afterwards.
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
            # Toward held. The main source of momentum is the movement to the midpoint (larger for narrow grazes).
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
    """Pick one touching same-type pair.

    The upper one (smaller y) takes top priority no matter what. At the same height, the side of travel (vx).
    """
    return _scan_merge_pair(bodies)[0]


def _scan_merge_pair(
    bodies: list[_BodyFruit],
) -> tuple[tuple[_BodyFruit, _BodyFruit] | None, float, float]:
    """The body of `_find_merge_pair`. Also returns the material for skipping as a by-product of the scan.

    Returns (the chosen pair, the gap of the nearest same-type pair, the speed of the fastest fruit on the board).
    The last two are used by `_MergeScan` to estimate 'after how many substeps contact can next happen'.
    The gap is inf when there is no same-type pair.

    pymunk position/velocity call into C through properties, so instead of rereading them
    every pair, each fruit is read once and cached locally before comparing.
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
    # The smallest gap of same-type pairs not in contact. Used to estimate skipping.
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
            # The top (smaller y) takes priority. At the same height, the direction of travel of the moving side.
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
    """Turn every fruit's position into a flat list alternating x, y."""
    snap: list[float] = []
    for item in bodies:
        pos = item.body.position
        snap.append(pos.x)
        snap.append(pos.y)
    return tuple(snap)


def _drifted(anchor: tuple[float, ...], current: tuple[float, ...]) -> bool:
    """Whether any fruit moved more than SLEEP_DRIFT from the position at the start of quiet.

    If the count changed, treat it as drift (start over).
    """
    if len(anchor) != len(current):
        return True
    for i in range(0, len(anchor), 2):
        if math.hypot(anchor[i] - current[i], anchor[i + 1] - current[i + 1]) > SLEEP_DRIFT:
            return True
    return False


def _all_quiet(bodies: list[_BodyFruit]) -> bool:
    """Whether every fruit is below the threshold in both speed and angular speed.

    Looks from the back. bodies are in insertion order, and the falling fruit (this drop) is at the end, so
    while it is falling False can be returned on the first one. Every fruit's velocity calls into C
    through pymunk properties, so the whole read is saved. It is just a predicate taking the AND over all fruits,
    so the order of looking does not change the result.
    """
    for item in reversed(bodies):
        body = item.body
        v = body.velocity
        if math.hypot(v.x, v.y) > SLEEP_VEL:
            return False
        if abs(body.angular_velocity) > SLEEP_ANG:
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
