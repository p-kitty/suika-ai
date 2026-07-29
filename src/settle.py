"""Wait for the board to stop after a drop."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .observe import Observation
from .vision.state import Fruit

# Movement below this is considered wobble (normalized coordinates px).
DEFAULT_STILL_PX = 2.5
# If quiet for this long throughout, consider it stopped.
DEFAULT_STILL_SEC = 0.7
# Give up if it has not moved by this long after the drop.
DEFAULT_TIMEOUT_SEC = 8.0
# Cap on the wait from the waiting fruit disappearing until the next one appears.
DEFAULT_HELD_TIMEOUT_SEC = 4.0
# Cap until 'the next move can be made', including waiting for ready.
DEFAULT_PLAYABLE_TIMEOUT_SEC = 12.0


def motion(previous: list[Fruit] | tuple[Fruit, ...], current: list[Fruit] | tuple[Fruit, ...]) -> float:
    """The max movement of fruits between consecutive frames.

    Pairs are matched one to one from the closest. Fruits present in only one frame add their radius
    as movement (appearing / disappearing also counts as 'movement').
    """
    if not previous and not current:
        return 0.0

    pairs = _pair(previous, current)
    matched_prev = {a for a, _ in pairs}
    matched_curr = {b for _, b in pairs}

    distances = [float(np.hypot(previous[a].x - current[b].x, previous[a].y - current[b].y)) for a, b in pairs]
    distances.extend(previous[i].radius for i in range(len(previous)) if i not in matched_prev)
    distances.extend(current[i].radius for i in range(len(current)) if i not in matched_curr)

    return max(distances) if distances else 0.0


def wait_settled(
    read: Callable[[], Observation],
    *,
    still_px: float = DEFAULT_STILL_PX,
    still_sec: float = DEFAULT_STILL_SEC,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> Observation:
    """Return an observation where the board's fruits have stopped."""
    deadline = time.monotonic() + timeout_sec
    quiet_since: float | None = None
    previous = read()

    while time.monotonic() < deadline:
        time.sleep(1 / 30)
        current = read()

        if current.blocked:
            return current

        moved = motion(previous.fruits, current.fruits)
        previous = current

        if moved <= still_px:
            if quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= still_sec:
                return current
        else:
            quiet_since = None

    return previous


def wait_ready(
    read: Callable[[], Observation],
    *,
    timeout_sec: float = DEFAULT_HELD_TIMEOUT_SEC,
) -> Observation:
    """Wait until the waiting fruit can be read again."""
    deadline = time.monotonic() + timeout_sec
    last = read()

    while time.monotonic() < deadline:
        if last.ready:
            return last
        if last.blocked:
            return last
        time.sleep(1 / 30)
        last = read()

    return last


def wait_playable(
    read: Callable[[], Observation],
    *,
    timeout_sec: float = DEFAULT_PLAYABLE_TIMEOUT_SEC,
) -> Observation:
    """Return an observation where the board has stopped and the waiting fruit is readable.

    After held appears the board may move again through a cascade, so right after not ready → ready
    right after, confirm stopping once more.
    """
    deadline = time.monotonic() + timeout_sec
    last = read()

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        last = wait_settled(read, timeout_sec=remaining)
        if last.blocked or last.ready:
            return last

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        last = wait_ready(read, timeout_sec=min(remaining, DEFAULT_HELD_TIMEOUT_SEC))
        if last.blocked:
            return last
        # Right after becoming ready, so settle again at the top of the loop.

    return last


def _pair(
    previous: list[Fruit] | tuple[Fruit, ...],
    current: list[Fruit] | tuple[Fruit, ...],
) -> list[tuple[int, int]]:
    candidates = []
    for a, left in enumerate(previous):
        for b, right in enumerate(current):
            distance = float(np.hypot(left.x - right.x, left.y - right.y))
            # Assumption: the same fruit's center does not move more than its radius.
            limit = max(left.radius, right.radius) * 2.5
            if distance <= limit:
                candidates.append((distance, a, b))

    pairs: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()

    for _, a, b in sorted(candidates):
        if a in used_a or b in used_b:
            continue
        pairs.append((a, b))
        used_a.add(a)
        used_b.add(b)

    return pairs
