"""操作まわりの単体テスト。画面やクリックは使わない。"""

from src.observe import Observation, clamp_drop_x, from_board
from src.settle import motion, wait_playable, wait_settled
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
    assert motion(before, after) >= 8.0


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
