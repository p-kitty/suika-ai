"""pymunk 落下物理の煙テスト。"""

from src.sim_physics import (
    _add_fruit,
    _build_space,
    _merge_pair,
    land_y,
    preview_land,
    simulate_drop,
)
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


def test_foreign_glance_kicks_target_quickly() -> None:
    # 1px のかすりでも横に弾く (60Hz 1 step だと高速落下が貫通して無反応だった)。
    orange_r = fruit_radius(4)
    cherry_r = fruit_radius(0)
    ox = 220.0
    orange = Fruit(
        type=4,
        x=ox,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    drop_x = ox - (orange_r + cherry_r - 1.0)
    after, _merges, _types = simulate_drop((orange,), 0, drop_x)
    moved = next(f for f in after if f.type == 4)
    assert abs(moved.x - ox) > 40.0


def test_center_drop_merge_stays_near_midpoint_column() -> None:
    r = fruit_radius(0)
    a = Fruit(type=0, x=200, y=NORMALIZED_HEIGHT - r, radius=r, confidence=90)
    after, merges, _types = simulate_drop((a,), 0, a.x)
    assert merges >= 1
    nxt = [f for f in after if f.type == 1]
    assert nxt
    # 真上合成は両中心の中点なので、列 x から大きくずれない。
    assert abs(nxt[0].x - a.x) < r


def test_side_contact_merge_happens() -> None:
    orange_r = fruit_radius(4)
    left = Fruit(
        type=4,
        x=180,
        y=NORMALIZED_HEIGHT - orange_r,
        radius=orange_r,
        confidence=90,
    )
    drop_x = left.x + orange_r * 1.85
    after, merges, _types = simulate_drop((left,), 4, drop_x)
    assert merges >= 1
    assert any(f.type == 5 for f in after)


def test_held_merge_pulls_toward_held() -> None:
    # held 合体は held 側へ引っ張られる。
    r = fruit_radius(4)
    ex = 200.0
    existing = Fruit(type=4, x=ex, y=NORMALIZED_HEIGHT - r, radius=r, confidence=90)
    for sign, frac in ((-1.0, 0.35), (1.0, 0.35)):
        drop_x = ex + sign * r * frac
        after, merges, _types = simulate_drop((existing,), 4, drop_x)
        assert merges >= 1
        apple = next(f for f in after if f.type == 5)
        mid = 0.5 * (ex + drop_x)
        assert (apple.x - mid) * sign > r * 0.15


def test_held_merge_pull_grows_with_side_offset() -> None:
    # ギリギリ側面ほど合体瞬間の横速度が大きい (移動量ベース)。

    from src.sim_physics import DT, _find_merge_pair

    r = fruit_radius(4)
    ex = 180.0
    speeds: list[float] = []
    for frac in (0.3, 1.4):
        existing = Fruit(type=4, x=ex, y=NORMALIZED_HEIGHT - r, radius=r, confidence=90)
        drop_x = ex + r * frac
        space, bodies = _build_space((existing,))
        held = _add_fruit(space, bodies, 4, drop_x, -r * 1.5)
        held.is_held_drop = True
        vx = 0.0
        for _ in range(500):
            pair = _find_merge_pair(bodies)
            if pair is not None:
                _merge_pair(space, bodies, pair[0], pair[1], [])
                vx = bodies[0].body.velocity.x
                break
            space.step(DT)
        speeds.append(vx)
    assert speeds[1] > speeds[0] * 1.5


def test_foreign_hit_clears_held_drop_flag() -> None:
    # 異種に触れた瞬間、held の横ひっぱ資格を失う。
    from src.sim_physics import _advance

    orange_r = fruit_radius(4)
    cherry_r = fruit_radius(0)
    ox = 220.0
    space, bodies = _build_space(
        (
            Fruit(
                type=4,
                x=ox,
                y=NORMALIZED_HEIGHT - orange_r,
                radius=orange_r,
                confidence=90,
            ),
        )
    )
    held = _add_fruit(space, bodies, 0, ox, -cherry_r * 1.5)
    held.is_held_drop = True
    for _ in range(120):
        _advance(space, bodies, [])
        if not held.is_held_drop:
            break
    assert not held.is_held_drop


def test_merge_without_held_flag_skips_side_pull() -> None:
    # is_held_drop が落ちたあとの合体は、同じ幾何でも横ひっぱが乗らない。
    from src.sim_physics import _find_merge_pair

    r = fruit_radius(4)
    y = NORMALIZED_HEIGHT - r
    space, bodies = _build_space(())
    _add_fruit(space, bodies, 4, 180.0, y, wake=False)
    held = _add_fruit(space, bodies, 4, 180.0 + r * 1.4, y - r * 0.2)
    held.body.velocity = (0.0, 400.0)
    held.is_held_drop = True
    pair = _find_merge_pair(bodies)
    assert pair is not None
    _merge_pair(space, bodies, pair[0], pair[1], [])
    vx_held = bodies[0].body.velocity.x

    space, bodies = _build_space(())
    _add_fruit(space, bodies, 4, 180.0, y, wake=False)
    other = _add_fruit(space, bodies, 4, 180.0 + r * 1.4, y - r * 0.2)
    other.body.velocity = (0.0, 400.0)
    other.is_held_drop = False
    pair = _find_merge_pair(bodies)
    assert pair is not None
    _merge_pair(space, bodies, pair[0], pair[1], [])
    vx_board = bodies[0].body.velocity.x

    assert abs(vx_held) > abs(vx_board) + 50.0


def test_merge_prefers_upper_over_velocity_direction() -> None:
    # 上があれば進行方向より上を優先。
    from src.sim_physics import _find_merge_pair

    r = fruit_radius(4)
    y = 300.0
    space, bodies = _build_space(())
    mid = _add_fruit(space, bodies, 4, 200.0, y, wake=False)
    up = _add_fruit(space, bodies, 4, 200.0, y - r * 1.9, wake=False)
    right = _add_fruit(space, bodies, 4, 200.0 + r * 1.9, y, wake=False)
    mid.body.velocity = (300.0, 0.0)
    pair = _find_merge_pair(bodies)
    assert pair is not None
    assert mid in pair and up in pair
    assert right not in pair


def test_merge_prefers_velocity_direction_when_same_height() -> None:
    # 同じ高さなら vx 方向の相手を優先。
    from src.sim_physics import _find_merge_pair

    r = fruit_radius(4)
    y = 300.0
    space, bodies = _build_space(())
    mid = _add_fruit(space, bodies, 4, 200.0, y, wake=False)
    left = _add_fruit(space, bodies, 4, 200.0 - r * 1.9, y, wake=False)
    right = _add_fruit(space, bodies, 4, 200.0 + r * 1.9, y, wake=False)
    mid.body.velocity = (300.0, 0.0)
    pair = _find_merge_pair(bodies)
    assert pair is not None
    assert mid in pair and right in pair
    assert left not in pair


def test_board_merge_cancels_opposing_velocity() -> None:
    # held 以外は運動量相殺のみ (反対速度ならほぼ止まる)。
    r = fruit_radius(4)
    y = NORMALIZED_HEIGHT - r
    space, bodies = _build_space(())
    a = _add_fruit(space, bodies, 4, 180.0, y, wake=False)
    b = _add_fruit(space, bodies, 4, 180.0 + r * 1.95, y, wake=False)
    a.body.velocity = (220.0, 0.0)
    b.body.velocity = (-220.0, 0.0)
    _merge_pair(space, bodies, a, b, [])
    assert len(bodies) == 1
    assert bodies[0].fruit_type == 5
    assert abs(bodies[0].body.velocity.x) < 15.0


def test_quiet_gate_rejects_slow_drift() -> None:
    # 速度閾値未満でも一方向にずれ続けると settled にしない。
    from src.sim_physics import (
        DT,
        SLEEP_DRIFT,
        SLEEP_FRAMES,
        SLEEP_VEL,
        _QuietGate,
        _add_fruit,
        _build_space,
    )

    space, bodies = _build_space(())
    space.gravity = (0.0, 0.0)
    r = fruit_radius(1)
    item = _add_fruit(space, bodies, 1, 200.0, NORMALIZED_HEIGHT - r, wake=False)
    # 閾値ギリギリ未満。旧判定 (速度だけ) だと SLEEP_FRAMES で止まる。
    creep = max(SLEEP_VEL * 0.9, 0.5)
    gate = _QuietGate()
    slept_at: int | None = None
    for step in range(SLEEP_FRAMES * 3):
        item.body.velocity = (creep, 0.0)
        space.step(DT)
        if gate.update(bodies):
            slept_at = step + 1
            break
    assert slept_at is None
    # ウィンドウ内の想定ずれがドリフト上限を超えること (係数の健全性)。
    assert creep * SLEEP_FRAMES * DT > SLEEP_DRIFT


def test_quiet_gate_accepts_true_rest() -> None:
    from src.sim_physics import DT, SLEEP_FRAMES, _QuietGate, _add_fruit, _build_space

    space, bodies = _build_space(())
    space.gravity = (0.0, 0.0)
    # 床接触の微小反発でドリフトしないよう、空中に静止させる。
    item = _add_fruit(space, bodies, 1, 200.0, 200.0, wake=False)
    item.body.velocity = (0.0, 0.0)
    item.body.angular_velocity = 0.0
    gate = _QuietGate()
    slept_at: int | None = None
    for step in range(SLEEP_FRAMES + 5):
        space.step(DT)
        if gate.update(bodies):
            slept_at = step + 1
            break
    assert slept_at == SLEEP_FRAMES


def test_preview_land_returns_finite() -> None:
    r = fruit_radius(0)
    x, y = preview_land((), 0, 200, r)
    assert 0 < x < 400
    assert y > 0
