"""Unit tests for aiming, dropping and returning to center. Uses neither the mouse nor windows."""

from __future__ import annotations

from contextlib import contextmanager
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
    """A simple board where mouse movement moves held_x."""

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
            # aim emits a step with the same sign as the error, so held increases in the positive direction.
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
    monkeypatch.setattr(control, "focus", lambda _title: None)
    monkeypatch.setattr(control.time, "sleep", lambda _sec: None)
    return fake


def test_aim_reaches_target_within_tolerance(world: FakeWorld) -> None:
    target = 260.0
    assert control.aim(target, world.read) is True
    assert abs(world.held_x - target) <= LOOK_CFG["look_tolerance"]
    assert world.moves  # something moved


def test_aim_accepts_crossing_near_target(world: FakeWorld) -> None:
    # Even with an error large enough to cross in one step, it succeeds if the crossing point is close.
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
    # When moving the view does not change held, at an edge and so on, drop there.
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
    """Env.step goes drop → settle wait → return to center, in that order, and restores the window at the end."""
    from src.env import Env
    from src.observe import Observation

    events: list[str] = []

    @contextmanager
    def fake_hidden(_title: str):
        events.append("hide")
        yield
        events.append("show")

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
    monkeypatch.setattr(control, "hidden", fake_hidden)
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
    assert events[0] == "playable"
    assert events[1] == "hide"
    assert events[-1] == "show"
    assert events[2:-1] == ["drop:250", "held_gone", "playable", "recenter"]
