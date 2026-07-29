"""狙い・落下・中央復帰の単体テスト。マウスも窓も使わない。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import control
from src.vision.normalized import NORMALIZED_WIDTH

LOOK_CFG = {
    "look_gain": 1.0,
    "look_tolerance": 8,
    "look_timeout_sec": 1.0,
    "look_max_step": 48,
    "recenter_tolerance": 14,
}


class FakeWorld:
    """マウス移動で held_x が動く簡易盤面。"""

    def __init__(self, held_x: float, *, blocked: bool = False) -> None:
        self.held_x = held_x
        self.blocked = blocked
        self.moves: list[int] = []
        self.clicks = 0
        self.clicked_at: float | None = None

    def read(self):
        obs = SimpleNamespace(
            held_x=None if self.blocked else self.held_x,
            blocked=self.blocked,
            ready=not self.blocked,
        )
        return obs, None

    def move_by(self, dx: int, dy: int = 0) -> None:
        self.moves.append(dx)
        if not self.blocked:
            # aim は誤差と同符号の step を出すので、正方向で held が増える。
            self.held_x += float(dx)

    def click(self) -> None:
        self.clicks += 1
        self.clicked_at = self.held_x


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    fake = FakeWorld(held_x=40.0)
    monkeypatch.setattr(control, "load", lambda: LOOK_CFG)
    monkeypatch.setattr(control, "move_by", fake.move_by)
    monkeypatch.setattr(control, "click", fake.click)
    monkeypatch.setattr(control.time, "sleep", lambda _sec: None)
    return fake


def test_aim_reaches_target_within_tolerance(world: FakeWorld) -> None:
    target = 260.0
    assert control.aim(target, world.read) is True
    assert abs(world.held_x - target) <= LOOK_CFG["look_tolerance"]
    assert world.moves  # 何か動かしている


def test_aim_accepts_crossing_near_target(world: FakeWorld) -> None:
    # 一歩でまたぐような大きな誤差でも、またいだ地点が近ければ成功扱い。
    world.held_x = 100.0
    target = 108.0
    assert control.aim(target, world.read) is True
    assert abs(world.held_x - target) <= control.CROSS_STOP


def test_drop_aims_then_clicks_at_target(world: FakeWorld) -> None:
    target = 300.0
    assert control.drop_column(target, read=world.read) is True
    assert world.clicks == 1
    assert world.clicked_at is not None
    assert abs(world.clicked_at - target) <= LOOK_CFG["look_tolerance"]


def test_recenter_brings_held_to_center(world: FakeWorld) -> None:
    world.held_x = 30.0
    assert control.recenter(world.read) is True
    center = NORMALIZED_WIDTH / 2
    assert abs(world.held_x - center) <= LOOK_CFG["recenter_tolerance"]


def test_aim_stops_when_held_cannot_move(world: FakeWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    # 端などで視線を動かしても held が変わらないときは、そこで落とす。
    monkeypatch.setattr(control, "move_by", lambda dx, dy=0: world.moves.append(dx))
    world.held_x = 10.0
    assert control.aim(200.0, world.read) is True
    assert abs(world.held_x - 10.0) < 1e-6
    assert len(world.moves) >= 3


def test_aim_fails_when_blocked(world: FakeWorld) -> None:
    world.blocked = True
    assert control.aim(200.0, world.read) is False
    assert world.moves == []


def test_step_drops_settles_and_recenters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env.step が 落とす → 静止待ち → 中央復帰 の順。"""
    from src.env import Env
    from src.observe import Observation

    events: list[str] = []

    ready = Observation(
        ready=True,
        blocked=False,
        fruits=(),
        held_type=0,
        held_x=80.0,
        next_type=1,
    )
    after = Observation(
        ready=True,
        blocked=False,
        fruits=(),
        held_type=0,
        held_x=NORMALIZED_WIDTH / 2,
        next_type=2,
    )

    env = Env()
    monkeypatch.setattr(env, "observe", lambda frame=None: ready)
    monkeypatch.setattr(
        control,
        "drop_column",
        lambda target, read: events.append(f"drop:{target:.0f}") or True,
    )
    monkeypatch.setattr(
        "src.env._wait_held_gone",
        lambda *_args, **_kwargs: events.append("held_gone"),
    )
    monkeypatch.setattr(
        "src.env.settle.wait_playable",
        lambda *_args, **_kwargs: events.append("playable") or after,
    )
    monkeypatch.setattr(
        control,
        "recenter",
        lambda read: events.append("recenter") or True,
    )

    result = env.step(250.0)

    assert result.info == "ok"
    assert result.done is False
    assert result.target_x == 250.0
    assert events == ["playable", "drop:250", "held_gone", "playable", "recenter"]
