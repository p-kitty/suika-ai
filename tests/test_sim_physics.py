"""pymunk 落下物理の煙テスト。"""

from src.sim_physics import land_y, preview_land, simulate_drop
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT
from src.vision.state import Fruit


def test_land_y_on_floor_when_empty() -> None:
    held_r = fruit_radius(0)
    assert abs(land_y((), 200, held_r) - (NORMALIZED_HEIGHT - held_r)) < 1e-6


def test_empty_drop_lands_on_floor() -> None:
    r = fruit_radius(0)
    after, merges, _types = simulate_drop((), 0, 200)
    assert merges == 0
    assert len(after) == 1
    assert after[0].type == 0
    assert abs(after[0].y - (NORMALIZED_HEIGHT - r)) < 3.0


def test_same_type_center_drop_merges() -> None:
    r = fruit_radius(0)
    a = Fruit(type=0, x=200, y=NORMALIZED_HEIGHT - r, radius=r, confidence=90)
    after, merges, _types = simulate_drop((a,), 0, a.x)
    assert merges >= 1
    assert any(f.type == 1 for f in after)


def test_foreign_hit_moves_both() -> None:
    orange_r = fruit_radius(4)
    orange = Fruit(
        type=4,
        x=220,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    after, _merges, _types = simulate_drop((orange,), 0, orange.x - orange_r * 0.35)
    moved = [f for f in after if f.type == 4]
    dropped = [f for f in after if f.type == 0]
    assert moved and dropped
    assert abs(moved[0].x - orange.x) > 8.0 or abs(
        dropped[0].x - (orange.x - orange_r * 0.35)
    ) > 8.0


def test_preview_land_returns_finite() -> None:
    r = fruit_radius(0)
    x, y = preview_land((), 0, 200, r)
    assert 0 < x < 400
    assert y > 0
