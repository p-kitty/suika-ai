"""落下後に盤面が止まるのを待つ。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace

from .observe import Observation
from .vision.state import Fruit

# 静止判定は raw_fruits (Tracker 平滑化前) を見る。
# 完全静止は待たず、遅い動きなら着手してよい。
# 正規化盤 (幅400) で 25px/s ≒ 0.4秒で約10px。検出ノイズより上、転がりより下。
DEFAULT_STILL_SPEED = 25.0
# この長さずっと遅ければ止まったとみなす。
DEFAULT_STILL_SEC = 0.4
# 落としてからここまで動かなければ諦める。
DEFAULT_TIMEOUT_SEC = 12.0
# 落下待ちが消えてから、次のが出るまでの待ち上限。
DEFAULT_HELD_TIMEOUT_SEC = 4.0
# ready 待ちを含めた「次の一手ができる」までの上限。
DEFAULT_PLAYABLE_TIMEOUT_SEC = 20.0
# 旧 API / テスト用。フレーム間 px。指定時は速度換算せずこの閾値を使う。
DEFAULT_STILL_PX = 1.5
# 出現・消失 1 個あたりの速度ペナルティ。フレーム間 px を dt で割ると
# 5px/frame ≈ 150px/s になり閾値を壊すので、点滅は低速扱いする。
UNMATCHED_SPEED = 12.0
# このフレーム数連続で速いときだけ静止タイマーを捨てる (1発の点滅は許す)。
NOISE_STREAK_RESET = 2


def motion(previous: list[Fruit] | tuple[Fruit, ...], current: list[Fruit] | tuple[Fruit, ...]) -> float:
    """前後フレームのフルーツの最大 |Δx|。

    落とす列が動いているかだけ見る。Y のバウンドや半径の検出ゆらぎは無視する。
    片方にしかいないフルーツは出現・消失として数える。
    """
    matched, unmatched = _motion_parts(previous, current)
    distances = matched + unmatched
    return max(distances) if distances else 0.0


def motion_speed(
    previous: list[Fruit] | tuple[Fruit, ...],
    current: list[Fruit] | tuple[Fruit, ...],
    dt: float,
) -> float:
    """静止判定用の横速度 (px/s)。

    マッチした組は |Δx|/dt。出現・消失はフレーム時間に依存させず定額にし、
    検出点滅で settle が永久に終わらないのを防ぐ。
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
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    abort: Callable[[], bool] | None = None,
    still_px: float | None = None,
) -> tuple[Observation, bool]:
    """盤面のフルーツが十分遅くなった観測を返す。

    既定は速度 (px/s) で判定する。still_px を渡したときだけフレーム間変位で見る
    (テスト用)。戻り値は (観測, 止まったか)。タイムアウトや中断なら False。
    """
    deadline = time.monotonic() + timeout_sec
    quiet_since: float | None = None
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
            if quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= still_sec:
                return current, True
        else:
            # 1フレームの検出点滅では quiet を捨てない。連続して速いときだけリセット。
            noise_streak += 1
            if noise_streak >= NOISE_STREAK_RESET:
                quiet_since = None

    return previous, False


def _motion_parts(
    previous: list[Fruit] | tuple[Fruit, ...],
    current: list[Fruit] | tuple[Fruit, ...],
) -> tuple[list[float], list[float]]:
    """(マッチ組の |Δx|, 出現・消失の擬似変位) を返す。"""
    if not previous and not current:
        return [], []

    pairs = _pair(previous, current)
    matched_prev = {a for a, _ in pairs}
    matched_curr = {b for _, b in pairs}

    # 列 (x) だけ。Y・半径の検出ゆらぎで settle を止めない。
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
    return replace(last, ready=False, blocked=False)


def _pair(
    previous: list[Fruit] | tuple[Fruit, ...],
    current: list[Fruit] | tuple[Fruit, ...],
) -> list[tuple[int, int]]:
    candidates = []
    for a, left in enumerate(previous):
        for b, right in enumerate(current):
            distance = math.hypot(left.x - right.x, left.y - right.y)
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
