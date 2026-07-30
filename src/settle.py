"""落下後に盤面が止まるのを待つ。"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .observe import Observation
from .vision.state import Fruit

# 静止判定は raw_fruits (Tracker 平滑化前) を見る。
# 検出ノイズで永遠に settle しないほどは厳しくしない。
DEFAULT_STILL_PX = 1.5
# この長さずっと静かなら止まったとみなす。
DEFAULT_STILL_SEC = 1.15
# 落としてからここまで動かなければ諦める。
DEFAULT_TIMEOUT_SEC = 12.0
# 落下待ちが消えてから、次のが出るまでの待ち上限。
DEFAULT_HELD_TIMEOUT_SEC = 4.0
# ready 待ちを含めた「次の一手ができる」までの上限。
DEFAULT_PLAYABLE_TIMEOUT_SEC = 20.0


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
    # 出現・消失も動きだが、半径まるごまだと検出点滅で永遠に settle しない。
    for i in range(len(previous)):
        if i not in matched_prev:
            distances.append(min(previous[i].radius, 10.0))
    for i in range(len(current)):
        if i not in matched_curr:
            distances.append(min(current[i].radius, 10.0))

    return max(distances) if distances else 0.0


def wait_settled(
    read: Callable[[], Observation],
    *,
    still_px: float = DEFAULT_STILL_PX,
    still_sec: float = DEFAULT_STILL_SEC,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    abort: Callable[[], bool] | None = None,
) -> tuple[Observation, bool]:
    """盤面のフルーツが止まった観測を返す。

    戻り値は (観測, 止まったか)。タイムアウトや中断なら最後の観測と False。
    """
    deadline = time.monotonic() + timeout_sec
    quiet_since: float | None = None
    previous = read()

    while time.monotonic() < deadline:
        if abort is not None and abort():
            return previous, False
        time.sleep(1 / 30)
        current = read()

        if current.blocked:
            return current, True

        moved = motion(previous.motion_fruits, current.motion_fruits)
        previous = current

        if moved <= still_px:
            if quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= still_sec:
                return current, True
        else:
            quiet_since = None

    return previous, False


def wait_ready(
    read: Callable[[], Observation],
    *,
    timeout_sec: float = DEFAULT_HELD_TIMEOUT_SEC,
    abort: Callable[[], bool] | None = None,
) -> Observation:
    """落下待ちフルーツが再び読めるまで待つ。"""
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
    """盤面が止まり、かつ落下待ちが読める観測を返す。

    held が出たあとに連鎖でまた動くことがあるので、not ready → ready の
    直後はもう一度静止を確認する。止まったと確認できないうちは ready でも返さない。
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
            # 動き続けたまま時間切れ。動いている盤面で手を決めない。
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
        # ready になった直後なので、ループ先頭で再度 settle する。

    # 止まりきらなかった / ready に戻れなかったときは、呼び出し側が
    # 「着手できない」と扱えるよう ready=False にする。
    if last.blocked or not last.ready:
        return last
    return Observation(
        ready=False,
        blocked=False,
        fruits=last.fruits,
        held_type=last.held_type,
        held_x=last.held_x,
        next_type=last.next_type,
        raw_fruits=last.raw_fruits,
    )


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
