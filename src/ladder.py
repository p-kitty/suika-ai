"""Detection of ladders (the shape that fires a corner big fruit up a staircase).

Put a peach (7) in the corner, a pear (6) next to it on the inside. On the shoulders of those two, an apple (5) and an orange (4).
Dropping an orange last cascades 4→5→6→7 and the corner peach becomes a pineapple.
The same holds from a corner pineapple onward; the staircase always goes down to the biggest drawable type (orange).

**For now it only detects and is not used for move selection.** It is not called from `policy.choose_x`,
and only tests/test_policy.py references it. What measurement has shown:

- Firing needs no guidance. Once a ladder is built, choose_x ties with the best of an exhaustive sweep over x
- Boards where it gets built do not appear (4 full rungs 12 times in 720 measured boards). This is where to intervene
- Without a filled floor the shape does not hold. The pear is pushed out like a wedge and self-destructs,
  and wherever you drop you get only one rung (15 points). A filled floor is a gate condition

Details in NOTES.md 'In progress: big draws and ladders after the floor fills'.
"""

from __future__ import annotations

from .penalties import MERGE_SLACK, is_wall_anchored, wall_gap
from .vision.classify import fruit_radius
from .vision.colors import SPAWN_MAX_TYPE
from .vision.state import Fruit

# The smallest type accepted as the base (peach).
MIN_ANCHOR_TYPE = 7
# The bottom rung of the ladder.
BASE_TYPE = SPAWN_MAX_TYPE
# The one below (dekopon). Two dekopons can make the orange rung, so
# it counts as a rung only when held/next are both dekopon.
FEED_TYPE = SPAWN_MAX_TYPE - 1


def find_anchor(
    fruits: list[Fruit] | tuple[Fruit, ...],
    sign: int,
) -> Fruit | None:
    """The base of the ladder. The biggest fruit on the big-side wall. None if below peach or away from the wall."""
    if not fruits:
        return None
    max_t = max(fruit.type for fruit in fruits)
    if max_t < MIN_ANCHOR_TYPE:
        return None
    best: Fruit | None = None
    for fruit in fruits:
        if fruit.type != max_t or not is_wall_anchored(fruit, sign):
            continue
        if best is None or wall_gap(fruit, sign) < wall_gap(best, sign):
            best = fruit
    return best


def _window(anchor: Fruit, sign: int) -> tuple[float, float]:
    """The horizontal band the ladder occupies. From slightly outside the base's center, to two pears' worth on the inside."""
    inner = anchor.radius + fruit_radius(anchor.type - 1) * 2.0 + MERGE_SLACK
    outer = anchor.radius * 0.5
    if sign > 0:
        return anchor.x - outer, anchor.x + inner
    return anchor.x - inner, anchor.x + outer


def _beside_anchor(anchor: Fruit, x: float, sign: int) -> bool:
    """Whether it is next to the base on the inside, not directly on top.

    The rung one smaller (the pear for a peach) goes alongside. Stacking it directly on top makes a shape that collapses.
    """
    return (x - anchor.x) * sign > anchor.radius * 0.5


def rungs(
    fruits: list[Fruit] | tuple[Fruit, ...],
    anchor: Fruit,
    sign: int,
) -> dict[int, Fruit]:
    """Rungs filled continuously downward from the base. Ends where it breaks.

    Rungs are inside the horizontal band, taken from the wall side, and not lower than the rung above.
    The pear is next to the peach (about the same y), so the floor radius difference is allowed.
    """
    lo, hi = _window(anchor, sign)
    found = {anchor.type: anchor}
    above = anchor
    for want in range(anchor.type - 1, FEED_TYPE - 1, -1):
        best: Fruit | None = None
        for fruit in fruits:
            if fruit.type != want or fruit is anchor:
                continue
            if not lo <= fruit.x <= hi:
                continue
            if fruit.y > above.y + above.radius:
                continue
            if want == anchor.type - 1 and not _beside_anchor(anchor, fruit.x, sign):
                continue
            if best is None or wall_gap(fruit, sign) < wall_gap(best, sign):
                best = fruit
        if best is None:
            break
        found[want] = best
        above = best
    return found
