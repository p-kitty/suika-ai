"""Wait for the board to stop after a drop."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace

from .observe import Observation
from .vision.state import Fruit

# The settle check looks at raw_fruits (before Tracker smoothing).
# Waiting is more stable than deciding x on a moving board. Do not compensate with lookahead.
# Instantaneous speed is set above detection noise (~1px@15fps ≈ 15px/s).
DEFAULT_STILL_SPEED = 25.0
# If slow for this long throughout, consider it stopped.
DEFAULT_STILL_SEC = 0.5
# Cap on sideways drift during the quiet window. Vibration noise is small, while one-directional creep accumulates.
DEFAULT_STILL_DRIFT = 3.0
# Give up if it has not moved by this long after the drop.
DEFAULT_TIMEOUT_SEC = 12.0
# Cap on the wait from the waiting fruit disappearing until the next one appears.
DEFAULT_HELD_TIMEOUT_SEC = 4.0
# Cap until 'the next move can be made', including waiting for ready.
DEFAULT_PLAYABLE_TIMEOUT_SEC = 20.0
# Speed penalty per appearance or disappearance. Dividing inter-frame px by dt
# gives 5px/frame ≈ 150px/s and breaks the threshold, so blinking is treated as slow.
UNMATCHED_SPEED = 12.0
# Only discard the settle timer when fast for this many consecutive frames (a single blink is allowed).
NOISE_STREAK_RESET = 2


def motion(previous: list[Fruit] | tuple[Fruit, ...], current: list[Fruit] | tuple[Fruit, ...]) -> float:
    """The max |Δx| of fruits between consecutive frames.

    Only looks at whether the drop column moves. Y bounces and radius detection wobble are ignored.
    Fruits present in only one frame count as appearing or disappearing.
    """
    matched, unmatched = _motion_parts(previous, current)
    distances = matched + unmatched
    return max(distances) if distances else 0.0


def motion_speed(
    previous: list[Fruit] | tuple[Fruit, ...],
    current: list[Fruit] | tuple[Fruit, ...],
    dt: float,
) -> float:
    """Sideways speed (px/s) for the settle check.

    Matched pairs are |Δx|/dt. Appearances and disappearances are a flat amount independent of frame time,
    so that detection blinking does not keep settle from ever finishing.
    """
    matched, unmatched = _motion_parts(previous, current)
    matched_speed = (max(matched) / max(dt, 1e-3)) if matched else 0.0
    unmatched_speed = len(unmatched) * UNMATCHED_SPEED
    return max(matched_speed, unmatched_speed)


def wait_settled(
    read: Callable[[], Observation],
    *,
    still_speed: float = DEFAULT_STILL_SPEED,
    still_sec: float = DEFAULT_STILL_SEC,
    still_drift: float = DEFAULT_STILL_DRIFT,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    abort: Callable[[], bool] | None = None,
    still_px: float | None = None,
) -> tuple[Observation, bool]:
    """Return an observation where the board's fruits are slow enough.

    By default judged by speed (px/s) and sideways drift during the quiet window.
    Slow one-directional creep is missed by speed alone, so drift makes it wait.
    Only when still_px is passed does it use inter-frame displacement (for tests).
    Returns (observation, stopped). False on timeout or interruption.
    """
    deadline = time.monotonic() + timeout_sec
    quiet_since: float | None = None
    quiet_anchor: tuple[Fruit, ...] | None = None
    noise_streak = 0
    previous = read()
    previous_t = time.monotonic()

    while time.monotonic() < deadline:
        if abort is not None and abort():
            return previous, False
        time.sleep(1 / 30)
        now = time.monotonic()
        current = read()

        if current.blocked:
            return current, True

        dt = max(now - previous_t, 1e-3)
        prev_fruits = previous.motion_fruits
        curr_fruits = current.motion_fruits
        previous = current
        previous_t = now

        if still_px is not None:
            quiet = motion(prev_fruits, curr_fruits) <= still_px
        else:
            quiet = motion_speed(prev_fruits, curr_fruits, dt) <= still_speed

        if quiet:
            noise_streak = 0
            if quiet_since is None or quiet_anchor is None:
                quiet_since = now
                quiet_anchor = tuple(curr_fruits)
            elif _max_x_drift(quiet_anchor, curr_fruits) > still_drift:
                # It keeps shifting in one direction. Start over from the position where it stopped.
                quiet_since = now
                quiet_anchor = tuple(curr_fruits)
            elif now - quiet_since >= still_sec:
                return current, True
        else:
            # Do not discard quiet on a one-frame detection blink. Reset only when fast consecutively.
            noise_streak += 1
            if noise_streak >= NOISE_STREAK_RESET:
                quiet_since = None
                quiet_anchor = None

    return previous, False


def _max_x_drift(
    anchor: list[Fruit] | tuple[Fruit, ...],
    current: list[Fruit] | tuple[Fruit, ...],
) -> float:
    """The max |Δx| from the position at the start of the quiet window.

    Only matched fruits are looked at. Appearances and disappearances are left to the blink tolerance on the speed side,
    so drift does not keep breaking quiet.
    """
    matched, _unmatched = _motion_parts(anchor, current)
    return max(matched) if matched else 0.0


def _motion_parts(
    previous: list[Fruit] | tuple[Fruit, ...],
    current: list[Fruit] | tuple[Fruit, ...],
) -> tuple[list[float], list[float]]:
    """Return (|Δx| of matched pairs, pseudo displacement of appearances and disappearances)."""
    if not previous and not current:
        return [], []

    pairs = _pair(previous, current)
    matched_prev = {a for a, _ in pairs}
    matched_curr = {b for _, b in pairs}

    # Column (x) only. Do not stop settle on Y or radius detection wobble.
    matched = [abs(float(previous[a].x - current[b].x)) for a, b in pairs]
    unmatched: list[float] = []
    for i in range(len(previous)):
        if i not in matched_prev:
            unmatched.append(min(previous[i].radius, 5.0))
    for i in range(len(current)):
        if i not in matched_curr:
            unmatched.append(min(current[i].radius, 5.0))
    return matched, unmatched


def wait_ready(
    read: Callable[[], Observation],
    *,
    timeout_sec: float = DEFAULT_HELD_TIMEOUT_SEC,
    abort: Callable[[], bool] | None = None,
) -> Observation:
    """Wait until the waiting fruit can be read again."""
    deadline = time.monotonic() + timeout_sec
    last = read()

    while time.monotonic() < deadline:
        if abort is not None and abort():
            return last
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
    abort: Callable[[], bool] | None = None,
) -> Observation:
    """Return an observation where the board has stopped and the waiting fruit is readable.

    After held appears the board may move again through a cascade, so right after not ready → ready
    confirm stopping once more. Do not return even when ready until stopping is confirmed.
    """
    deadline = time.monotonic() + timeout_sec
    last = read()

    while time.monotonic() < deadline:
        if abort is not None and abort():
            return last

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        last, settled = wait_settled(read, timeout_sec=remaining, abort=abort)
        if abort is not None and abort():
            return last
        if last.blocked:
            return last
        if settled and last.ready:
            return last
        if not settled:
            # Timed out while still moving. Do not decide a move on a moving board.
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        last = wait_ready(
            read,
            timeout_sec=min(remaining, DEFAULT_HELD_TIMEOUT_SEC),
            abort=abort,
        )
        if abort is not None and abort():
            return last
        if last.blocked:
            return last
        # Right after becoming ready, so settle again at the top of the loop.

    # When it did not fully stop / could not get back to ready, set ready=False
    # so the caller can treat it as 'cannot move'.
    if last.blocked or not last.ready:
        return last
    return replace(last, ready=False, blocked=False)


def _pair(
    previous: list[Fruit] | tuple[Fruit, ...],
    current: list[Fruit] | tuple[Fruit, ...],
) -> list[tuple[int, int]]:
    candidates = []
    for a, left in enumerate(previous):
        for b, right in enumerate(current):
            distance = math.hypot(left.x - right.x, left.y - right.y)
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
