"""Unit tests for aiming, dropping and returning to center. Uses neither the mouse nor windows."""

from __future__ import annotations

import pytest

from src import control
from src.observe import Observation
from src.vision.normalized import NORMALIZED_WIDTH


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    fake = FakeWorld(held_x=40.0)
    # In tests, move fast and big so it converges.
    monkeypatch.setattr(control, "LOOK_GAIN", 1.0)
    monkeypatch.setattr(control, "LOOK_TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(control, "move_by", fake.move_by)
    monkeypatch.setattr(control, "click", fake.click)
    monkeypatch.setattr(control.time, "sleep", lambda _sec: None)
    return fake


class FakeWorld:
    """A simple board where mouse movement moves held_x."""

    def __init__(self, held_x: float, *, blocked: bool = False) -> None:
        self.held_x = held_x
        self.blocked = blocked
        self.moves: list[int] = []
        self.clicks = 0
        self.clicked_at: float | None = None

    def read(self) -> tuple[Observation, None]:
        obs = Observation(
            ready=not self.blocked,
            blocked=self.blocked,
            fruits=(),
            held_type=None,
            held_x=None if self.blocked else self.held_x,
            next_type=None,
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


def test_aim_reaches_target_within_tolerance(world: FakeWorld) -> None:
    target = 260.0
    assert control.aim(target, world.read) is True
    assert abs(world.held_x - target) <= control.LOOK_TOLERANCE
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
    assert abs(world.clicked_at - target) <= control.LOOK_TOLERANCE


def test_recenter_returns_to_center_from_edge(world: FakeWorld) -> None:
    world.held_x = 20.0
    assert control.recenter(world.read) is True
    assert abs(world.held_x - control.RECENTER_X) <= control.RECENTER_TOLERANCE


def test_recenter_skips_when_already_centered(world: FakeWorld) -> None:
    world.held_x = control.RECENTER_X
    assert control.recenter(world.read) is True
    assert world.moves == []
    assert world.held_x == control.RECENTER_X


def test_aim_stops_when_held_cannot_move(world: FakeWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    # When moving the view does not change held, at an edge and so on, drop there.
    monkeypatch.setattr(control, "move_by", lambda dx, dy=0: world.moves.append(dx))
    world.held_x = 10.0
    assert control.aim(200.0, world.read) is True
    assert abs(world.held_x - 10.0) < 1e-6
    assert len(world.moves) >= control.STALL_MOVES


def test_aim_stops_early_near_right_edge(world: FakeWorld) -> None:
    # When aiming at the right edge and held has reached the wall, do not swing the view for the last few px.
    world.held_x = NORMALIZED_WIDTH - 30.0
    target = NORMALIZED_WIDTH - 12.0
    assert control.aim(target, world.read) is True
    assert world.moves == []
    assert abs(world.held_x - (NORMALIZED_WIDTH - 30.0)) < 1e-6


def test_aim_stops_early_near_left_edge(world: FakeWorld) -> None:
    world.held_x = 28.0
    target = 10.0
    assert control.aim(target, world.read) is True
    assert world.moves == []


def test_aim_does_not_stop_early_for_inward_target_from_edge(world: FakeWorld) -> None:
    # When, after staying near the right edge on the first move, aiming at the inner column next to the edge.
    # Do not stop early for an aim inside EDGE_BAND that is not the wall itself
    # (otherwise it lands on the shoulder and gets knocked to the far side).
    world.held_x = NORMALIZED_WIDTH - 18.0
    target = NORMALIZED_WIDTH - 55.0
    assert control.aim(target, world.read) is True
    assert world.moves
    assert abs(world.held_x - target) <= control.LOOK_TOLERANCE


def test_aim_fails_when_blocked(world: FakeWorld) -> None:
    world.blocked = True
    assert control.aim(200.0, world.read) is False
    assert world.moves == []


def test_aim_aborts_without_finishing(world: FakeWorld) -> None:
    calls = {"n": 0}

    def abort() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    world.held_x = 40.0
    assert control.aim(300.0, world.read, abort=abort) is False
    assert abs(world.held_x - 300.0) > control.LOOK_TOLERANCE


def test_drop_skips_click_when_aborted(world: FakeWorld) -> None:
    assert control.drop_column(300.0, read=world.read, abort=lambda: True) is False
    assert world.clicks == 0
    assert world.moves == []


def test_step_drops_settles_and_recenters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env.step goes drop → settle wait → return to center, in that order."""
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
        lambda target, read, abort=None: events.append(f"drop:{target:.0f}") or True,
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
        lambda read, abort=None: events.append("recenter") or True,
    )

    result = env.step(250.0)

    assert result.info == "ok"
    assert result.done is False
    assert result.target_x == 250.0
    assert events == ["playable", "drop:250", "held_gone", "playable", "recenter"]
