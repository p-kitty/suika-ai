"""落下後に盤面が止まるのを待つ。"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .observe import Observation
from .vision.state import Fruit

# これ未満の移動は揺らぎとみなす (正規化座標 px)。
DEFAULT_STILL_PX = 2.5
# この長さずっと静かなら止まったとみなす。
DEFAULT_STILL_SEC = 0.45
# 落としてからここまで動かなければ諦める。
DEFAULT_TIMEOUT_SEC = 8.0
# 落下待ちが消えてから、次のが出るまでの待ち上限。
DEFAULT_HELD_TIMEOUT_SEC = 4.0


def motion(previous: list[Fruit] | tuple[Fruit, ...], current: list[Fruit] | tuple[Fruit, ...]) -> float:
    """前後フレームのフルーツの最大移動量。

    近い組から一対一で対応づける。片方にしかいないフルーツは、その半径を
    移動量として足す (出現・消失も「動き」とみなす)。
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
    """盤面のフルーツが止まった観測を返す。"""
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
    """落下待ちフルーツが再び読めるまで待つ。"""
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


def _pair(
    previous: list[Fruit] | tuple[Fruit, ...],
    current: list[Fruit] | tuple[Fruit, ...],
) -> list[tuple[int, int]]:
    candidates = []
    for a, left in enumerate(previous):
        for b, right in enumerate(current):
            distance = float(np.hypot(left.x - right.x, left.y - right.y))
            # 同じフルーツなら中心は半径ぶんより大きくは動かない、という仮定。
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
