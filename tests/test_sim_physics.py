"""pymunk 落下物理の煙テスト。"""

import math

from src.sim_physics import (
    _add_fruit,
    _build_space,
    _merge_pair,
    land_y,
    landed_xy,
    preview_land,
    simulate_drop,
    simulate_drop_held,
)
from src.vision.classify import fruit_radius
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
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


def _melon_pineapple_wedge() -> tuple[Fruit, Fruit, float]:
    """メロンとパインを床で接するように並べ、継ぎ目の真上を返す。

    実機の実測: 継ぎ目にグレープを落とすと谷を抜けて床まで届くが、
    デコポンは谷に挟まって浮いたまま止まる (WATERMELON_RADIUS_RATIO の
    校正がずれるとこの境目が簡単に崩れるので、その回帰止め)。
    """
    melon_r = fruit_radius(9)
    pine_r = fruit_radius(8)
    melon = Fruit(type=9, x=150, y=NORMALIZED_HEIGHT - melon_r, radius=melon_r, confidence=90)
    pine = Fruit(
        type=8,
        x=150 + melon_r + pine_r,
        y=NORMALIZED_HEIGHT - pine_r,
        radius=pine_r,
        confidence=90,
    )
    return melon, pine, 150 + melon_r


def test_grape_falls_through_melon_pineapple_wedge() -> None:
    melon, pine, seam_x = _melon_pineapple_wedge()
    grape_r = fruit_radius(2)
    after, _merges, _types = simulate_drop((melon, pine), 2, seam_x)
    grape = next(f for f in after if f.type == 2)
    assert abs(grape.y - (NORMALIZED_HEIGHT - grape_r)) < 3.0


def test_dekopon_wedges_above_melon_pineapple_seam() -> None:
    melon, pine, seam_x = _melon_pineapple_wedge()
    dekopon_r = fruit_radius(3)
    after, _merges, _types = simulate_drop((melon, pine), 3, seam_x)
    dekopon = next(f for f in after if f.type == 3)
    assert dekopon.y - (NORMALIZED_HEIGHT - dekopon_r) < -10.0


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


def test_strawberry_at_left_wall_keeps_size_order_with_cherry_and_grape() -> None:
    # 左端に隙間なくチェリー、その右にブドウ。左端ぎりぎりにイチゴを落としても、
    # チェリーを弾き飛ばさずチェリーとブドウの間に収まってほしい (大小順を保つ)。
    cherry_r = fruit_radius(0)
    straw_r = fruit_radius(1)
    grape_r = fruit_radius(2)

    cherry = Fruit(
        type=0, x=cherry_r, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90
    )
    grape = Fruit(
        type=2,
        x=cherry.x + cherry_r + grape_r,
        y=NORMALIZED_HEIGHT - grape_r,
        radius=grape_r,
        confidence=90,
    )

    after, merges, _types = simulate_drop((cherry, grape), 1, straw_r)
    assert merges == 0
    by_x = sorted(after, key=lambda f: f.x)
    assert [f.type for f in by_x] == [0, 1, 2]
    # チェリーが弾き飛ばされて盤の反対側近くまで飛んでいないこと。
    moved_cherry = next(f for f in after if f.type == 0)
    assert abs(moved_cherry.x - cherry.x) < 40.0


def test_simulate_drop_held_true_when_dropped_fruit_merges() -> None:
    r = fruit_radius(0)
    a = Fruit(type=0, x=200, y=NORMALIZED_HEIGHT - r, radius=r, confidence=90)
    _after, merges, _types, held_merged = simulate_drop_held((a,), 0, a.x)
    assert merges >= 1
    assert held_merged


def test_simulate_drop_held_true_after_glancing_off_foreign_fruit() -> None:
    # held が異種 (梨) をかすってから同種 (グレープ) に合体しても、
    # is_held_drop は異種接触で消えるが is_held_lineage は消えずに拾える。
    pear_r = fruit_radius(6)
    grape_r = fruit_radius(2)
    left_pear = Fruit(type=6, x=160, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    gy = NORMALIZED_HEIGHT - grape_r
    dy = gy - left_pear.y
    gx = left_pear.x + math.sqrt((pear_r + grape_r) ** 2 - dy * dy)
    existing_grape = Fruit(type=2, x=gx, y=gy, radius=grape_r, confidence=90)

    _after, merges, _types, held_merged = simulate_drop_held(
        (left_pear, existing_grape), 2, existing_grape.x
    )
    assert merges >= 1
    assert held_merged


def test_simulate_drop_held_false_for_unrelated_merge() -> None:
    # held と無関係な既存ペアが (held の着地とは離れた場所で) 合体しても、
    # held_merged は False のまま (無関係な合体まで拾わない)。
    r = fruit_radius(0)
    a = Fruit(type=0, x=40, y=NORMALIZED_HEIGHT - r, radius=r, confidence=90)
    b = Fruit(type=0, x=40 + r * 1.9, y=NORMALIZED_HEIGHT - r, radius=r, confidence=90)
    held_r = fruit_radius(4)
    far_x = NORMALIZED_WIDTH - held_r - 4
    _after, merges, _types, held_merged = simulate_drop_held((a, b), 4, far_x)
    assert merges >= 1
    assert not held_merged


def test_landed_xy_uses_real_position_when_only_unrelated_pair_merged() -> None:
    # 無関係なペアが合体しても、held 自身が生き残っていれば実際の静止位置を返す。
    # merges で切っていた頃はここで幾何推定を返し、それを受け取る減点
    # (_bury_block_penalty など) が嘘の座標で動いていた。
    cherry_r = fruit_radius(0)
    melon_r = fruit_radius(9)
    orange_r = fruit_radius(4)

    a = Fruit(type=0, x=30, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    b = Fruit(
        type=0,
        x=30 + cherry_r * 1.9,
        y=NORMALIZED_HEIGHT - cherry_r,
        radius=cherry_r,
        confidence=90,
    )
    # held が肩に当たって大きく転がる相手。
    melon = Fruit(type=9, x=300, y=NORMALIZED_HEIGHT - melon_r, radius=melon_r, confidence=90)
    fruits = (a, b, melon)
    drop_x = melon.x - melon_r * 0.75

    after, merges, _types, held_merged = simulate_drop_held(fruits, 4, drop_x)
    assert merges >= 1
    assert not held_merged

    land_x, land_y_ = landed_xy(fruits, after, 4, drop_x, orange_r, held_merged)
    orange = next(f for f in after if f.type == 4)
    assert abs(land_x - orange.x) < 1e-6
    assert abs(land_y_ - orange.y) < 1e-6
    # 幾何推定 (旧ゲートが返していた値) とは大きく違うこと。
    est_y = land_y(fruits, drop_x, orange_r)
    assert math.hypot(land_x - drop_x, land_y_ - est_y) > 50.0
