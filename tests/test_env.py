"""操作まわりの単体テスト。画面やクリックは使わない。"""

from src.observe import Observation, clamp_drop_x, from_board
from src.settle import motion, motion_speed, wait_playable, wait_settled
from src.vision.board import BoardResult
from src.vision.classify import ClassifyResult
from src.vision.held import HeldResult
from src.vision.next import NextResult
from src.vision.normalized import NORMALIZED_WIDTH
from src.vision.state import Fruit


def _obs(*, x: float, ready: bool = True) -> Observation:
    return Observation(
        ready=ready,
        blocked=False,
        fruits=(Fruit(type=0, x=x, y=20, radius=5, confidence=90),),
        held_type=0 if ready else None,
        held_x=100.0 if ready else None,
        next_type=None,
    )


def test_clamp_keeps_fruit_inside_walls() -> None:
    # cherry の半径は roughly 12px。端に落とそうとしても壁の内側に戻る。
    assert clamp_drop_x(-10, fruit_type=0) > 0
    assert clamp_drop_x(NORMALIZED_WIDTH + 10, fruit_type=0) < NORMALIZED_WIDTH
    assert clamp_drop_x(200, fruit_type=0) == 200


def test_observation_ready_needs_held() -> None:
    board = BoardResult(
        normalized=None,
        corners=None,
        found=True,
        fruits=[],
        held_fruit=HeldResult(fruit=None),
        next_fruit=NextResult(fruit=None),
    )
    assert not from_board(board).ready

    board.held_fruit = HeldResult(
        fruit=ClassifyResult(type=0, confidence=90),
        x=120,
        y=-60,
        radius=12,
        radius_ratio=0.03,
    )
    obs = from_board(board)
    assert obs.ready
    assert obs.held_type == 0
    assert obs.held_x == 120


def test_blocked_board_is_not_ready() -> None:
    board = BoardResult(normalized=None, corners=None, found=True, blocked=True)
    obs = from_board(board)
    assert obs.blocked
    assert not obs.ready


def test_motion_zero_when_identical() -> None:
    fruits = [Fruit(type=0, x=10, y=20, radius=5, confidence=90)]
    assert motion(fruits, fruits) == 0.0


def test_motion_tracks_shift() -> None:
    before = [Fruit(type=0, x=10, y=20, radius=5, confidence=90)]
    after = [Fruit(type=0, x=14, y=20, radius=5, confidence=90)]
    assert abs(motion(before, after) - 4.0) < 1e-6


def test_motion_treats_new_fruit_as_movement() -> None:
    before = [Fruit(type=0, x=10, y=20, radius=5, confidence=90)]
    after = [
        Fruit(type=0, x=10, y=20, radius=5, confidence=90),
        Fruit(type=1, x=50, y=80, radius=8, confidence=90),
    ]
    assert motion(before, after) >= 5.0


def test_motion_speed_does_not_treat_blink_as_roll() -> None:
    # 5px の点滅を dt=1/30 で割ると 150px/s になり、25px/s 閾値を壊す。
    # 速度判定では出現・消失を定額にし、点滅だけで settle 不能にしない。
    still = [Fruit(type=0, x=10, y=20, radius=5, confidence=90)]
    blink = [
        Fruit(type=0, x=10, y=20, radius=5, confidence=90),
        Fruit(type=1, x=50, y=80, radius=8, confidence=90),
    ]
    assert motion_speed(still, blink, dt=1 / 30) <= 25.0
    rolling = [Fruit(type=0, x=14, y=20, radius=5, confidence=90)]
    assert motion_speed(still, rolling, dt=1 / 30) > 25.0


def test_motion_ignores_y_and_radius_jitter() -> None:
    # 列が同じなら Y バウンドや半径ゆらぎは動きに数えない。
    before = [Fruit(type=0, x=10, y=20, radius=5, confidence=90)]
    after = [Fruit(type=0, x=10, y=40, radius=9, confidence=90)]
    assert motion(before, after) == 0.0
    assert motion_speed(before, after, dt=1 / 30) == 0.0


def test_wait_settled_tolerates_detection_blink(monkeypatch) -> None:
    # ほぼ静止＋時々の検出点滅でも settle できる。
    monkeypatch.setattr("src.settle.time.sleep", lambda _sec: None)
    now = {"t": 0.0}
    monkeypatch.setattr("src.settle.time.monotonic", lambda: now["t"])
    n = {"i": 0}

    def read() -> Observation:
        now["t"] += 1 / 30
        n["i"] += 1
        fruits = [Fruit(type=0, x=10.0, y=20, radius=5, confidence=90)]
        if n["i"] % 3 == 0:
            fruits.append(Fruit(type=1, x=50, y=80, radius=8, confidence=90))
        return Observation(
            ready=True,
            blocked=False,
            fruits=tuple(fruits),
            held_type=0,
            held_x=100.0,
            next_type=None,
            raw_fruits=tuple(fruits),
        )

    obs, settled = wait_settled(read, still_speed=25.0, still_sec=0.2, timeout_sec=2.0)
    assert settled
    assert obs.ready


def test_wait_settled_returns_early_on_abort(monkeypatch) -> None:
    monkeypatch.setattr("src.settle.time.sleep", lambda _sec: None)
    obs = _obs(x=10.0)
    calls = {"n": 0}

    def abort() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    result, settled = wait_settled(lambda: obs, timeout_sec=5.0, abort=abort)
    assert result is obs
    assert not settled
    assert calls["n"] >= 2


def test_wait_settled_reports_timeout_while_moving(monkeypatch) -> None:
    monkeypatch.setattr("src.settle.time.sleep", lambda _sec: None)
    now = {"t": 0.0}
    monkeypatch.setattr("src.settle.time.monotonic", lambda: now["t"])

    xs = iter([10.0, 14.0, 18.0, 22.0])

    def read() -> Observation:
        now["t"] += 0.05
        return _obs(x=next(xs, 22.0))

    obs, settled = wait_settled(read, still_px=1.0, still_sec=0.5, timeout_sec=0.2)
    assert not settled
    assert obs.ready


def test_wait_settled_true_after_quiet(monkeypatch) -> None:
    monkeypatch.setattr("src.settle.time.sleep", lambda _sec: None)
    now = {"t": 0.0}
    monkeypatch.setattr("src.settle.time.monotonic", lambda: now["t"])

    def read() -> Observation:
        now["t"] += 0.05
        return _obs(x=10.0)

    obs, settled = wait_settled(read, still_px=1.0, still_sec=0.2, timeout_sec=2.0)
    assert settled
    assert obs.fruits[0].x == 10.0


def test_wait_settled_allows_slow_creep(monkeypatch) -> None:
    # 完全静止でなく、遅い動き (〜15px/s) なら着手してよい。
    monkeypatch.setattr("src.settle.time.sleep", lambda _sec: None)
    now = {"t": 0.0}
    monkeypatch.setattr("src.settle.time.monotonic", lambda: now["t"])

    def read() -> Observation:
        now["t"] += 0.05
        return _obs(x=10.0 + now["t"] * 15.0)

    obs, settled = wait_settled(
        read, still_speed=25.0, still_sec=0.2, timeout_sec=2.0
    )
    assert settled
    assert obs.ready


def test_wait_playable_does_not_return_ready_while_moving(monkeypatch) -> None:
    # 旧挙動: settle タイムアウトでも ready なら返していた。
    # 動いている盤面では ready=False にして着手させない。
    monkeypatch.setattr("src.settle.time.sleep", lambda _sec: None)
    now = {"t": 0.0}
    monkeypatch.setattr("src.settle.time.monotonic", lambda: now["t"])

    def read() -> Observation:
        x = 10.0 + now["t"] * 20.0
        now["t"] += 0.05
        return _obs(x=x)

    obs = wait_playable(read, timeout_sec=0.3)
    assert not obs.ready
    assert not obs.blocked


def test_wait_playable_returns_when_still_and_ready(monkeypatch) -> None:
    monkeypatch.setattr("src.settle.time.sleep", lambda _sec: None)
    now = {"t": 0.0}
    monkeypatch.setattr("src.settle.time.monotonic", lambda: now["t"])

    def read() -> Observation:
        now["t"] += 0.05
        return _obs(x=10.0)

    obs = wait_playable(read, timeout_sec=2.0)
    assert obs.ready
    assert obs.fruits[0].x == 10.0


def test_wait_settled_uses_raw_fruits_not_smoothed(monkeypatch) -> None:
    # Tracker 相当で fruits は止まって見えても、raw が動いていれば未静止。
    monkeypatch.setattr("src.settle.time.sleep", lambda _sec: None)
    now = {"t": 0.0}
    monkeypatch.setattr("src.settle.time.monotonic", lambda: now["t"])

    def read() -> Observation:
        now["t"] += 0.05
        raw_x = 10.0 + now["t"] * 40.0
        return Observation(
            ready=True,
            blocked=False,
            fruits=(Fruit(type=0, x=10.0, y=20, radius=5, confidence=90),),
            held_type=0,
            held_x=100.0,
            next_type=None,
            raw_fruits=(Fruit(type=0, x=raw_x, y=20, radius=5, confidence=90),),
        )

    obs, settled = wait_settled(read, still_px=1.0, still_sec=0.2, timeout_sec=0.35)
    assert not settled
    assert obs.ready


def test_step_chooses_after_settle(monkeypatch) -> None:
    # 静止後の観測で choose し、その列で落とす。動いている盤を読まない。
    from src import control
    from src.env import Env

    ready = _obs(x=10.0)
    chosen: list[float] = []

    def choose(obs: Observation) -> float:
        chosen.append(obs.fruits[0].x)
        return 250.0

    env = Env()
    monkeypatch.setattr(env, "observe", lambda frame=None: ready)
    monkeypatch.setattr("src.env.settle.wait_playable", lambda *_a, **_k: ready)
    monkeypatch.setattr(
        control,
        "drop_column",
        lambda target, read, abort=None: chosen.append(target) or True,
    )
    monkeypatch.setattr("src.env._wait_held_gone", lambda *_a, **_k: None)
    monkeypatch.setattr(control, "recenter", lambda *_a, **_k: True)

    result = env.step(abort=None, choose=choose)
    assert result.info == "ok"
    assert result.target_x == 250.0
    assert chosen[0] == 10.0
    assert chosen[1] == 250.0


def test_step_timeout_does_not_end_episode(monkeypatch) -> None:
    from src.env import Env

    ready = _obs(x=10.0)
    env = Env()
    monkeypatch.setattr(env, "observe", lambda frame=None: ready)
    monkeypatch.setattr(
        "src.env.settle.wait_playable",
        lambda *_args, **_kwargs: Observation(
            ready=False,
            blocked=False,
            fruits=ready.fruits,
            held_type=0,
            held_x=100.0,
            next_type=None,
        ),
    )

    result = env.step(200.0)
    assert result.info == "not settled"
    assert result.done is False


def test_step_after_drop_timeout_does_not_end_episode(monkeypatch) -> None:
    from src import control
    from src.env import Env

    ready = _obs(x=10.0)
    unsettled = Observation(
        ready=False,
        blocked=False,
        fruits=ready.fruits,
        held_type=0,
        held_x=100.0,
        next_type=None,
    )
    calls = {"n": 0}

    def playable(*_args, **_kwargs):
        calls["n"] += 1
        return ready if calls["n"] == 1 else unsettled

    env = Env()
    monkeypatch.setattr(env, "observe", lambda frame=None: ready)
    monkeypatch.setattr("src.env.settle.wait_playable", playable)
    monkeypatch.setattr(control, "drop_column", lambda *_a, **_k: True)
    monkeypatch.setattr("src.env._wait_held_gone", lambda *_a, **_k: None)

    result = env.step(200.0)
    assert result.info == "timeout"
    assert result.done is False
