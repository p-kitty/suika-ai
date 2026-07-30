"""操作まわりの単体テスト。画面やクリックは使わない。"""

from src.observe import clamp_drop_x, from_board
from src.settle import motion
from src.vision.board import BoardResult
from src.vision.classify import ClassifyResult
from src.vision.held import HeldResult
from src.vision.next import NextResult
from src.vision.normalized import NORMALIZED_WIDTH
from src.vision.state import Fruit


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
    from src.observe import Observation
    from src.settle import wait_settled

    monkeypatch.setattr("src.settle.time.sleep", lambda _sec: None)
    obs = Observation(
        ready=True,
        blocked=False,
        fruits=(Fruit(type=0, x=10, y=20, radius=5, confidence=90),),
        held_type=0,
        held_x=100.0,
        next_type=None,
    )
    calls = {"n": 0}

    def abort() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    result = wait_settled(lambda: obs, timeout_sec=5.0, abort=abort)
    assert result is obs
    assert calls["n"] >= 2
