"""方策の単体テスト。画面は使わない。"""

from src.observe import Observation
from src.policy import (
    _after_drop,
    _chain_center_gap,
    _ideal_x,
    _land_y,
    _preview_land,
    _radius,
    _score,
    choose_x,
)
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit


def _obs(*, held_type: int, fruits: tuple[Fruit, ...] = (), next_type: int | None = None) -> Observation:
    return Observation(
        ready=True,
        blocked=False,
        fruits=fruits,
        held_type=held_type,
        held_x=NORMALIZED_WIDTH / 2,
        next_type=next_type,
    )


def test_empty_board_drops_near_ideal_for_size() -> None:
    # 空盤ではサイズ順の ideal 付近 (cherry は右寄り)。
    x = choose_x(_obs(held_type=0))
    assert abs(x - _ideal_x(0)) < 40


def test_prefers_same_type_over_empty_low_column() -> None:
    # 右に同種、左は空きだが床だけ。同種の上／そばを選ぶ。
    cherry_r = _radius(0)
    same = Fruit(type=0, x=280, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(same,)))
    # 真上か横付けのどちらか。左の空き床よりは同種側。
    assert abs(x - same.x) < cherry_r * 3
    assert x > 200


def test_avoids_dangerous_tall_stack() -> None:
    # 左は天井近くまで積んである。右は低い同種が無い空き。低い方へ。
    big_r = _radius(5)
    tall = Fruit(type=5, x=80, y=60 + big_r, radius=big_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(tall,)))
    assert abs(x - tall.x) > 80
    assert x >= NORMALIZED_WIDTH / 2


def test_land_y_on_floor_when_empty() -> None:
    held_r = _radius(0)
    assert abs(_land_y((), 200, held_r) - (NORMALIZED_HEIGHT - held_r)) < 1e-6


def test_land_y_rests_on_fruit() -> None:
    held_r = _radius(0)
    fruit = Fruit(type=1, x=200, y=400, radius=20, confidence=90)
    land = _land_y((fruit,), 200, held_r)
    assert abs(land - (fruit.y - fruit.radius - held_r)) < 1e-6


def test_prefers_merge_that_lowers_stack() -> None:
    # 右に同種2つが少し離れてあり、間に落とすと合成。左は空き床。合成列を選ぶ。
    cherry_r = _radius(0)
    floor_y = NORMALIZED_HEIGHT - cherry_r
    # 接触しない間隔 (2r + CONTACT より広く、3つ目で両方に届く距離)。
    a = Fruit(type=0, x=250, y=floor_y, radius=cherry_r, confidence=90)
    b = Fruit(type=0, x=250 + 2 * cherry_r + 8, y=floor_y, radius=cherry_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(a, b)))
    assert x > 200
    after, merges = _after_drop(_obs(held_type=0, fruits=(a, b)), x)
    assert merges >= 1
    assert abs(x - (a.x + b.x) / 2) < cherry_r * 4


def test_does_not_bury_same_type_under_different() -> None:
    # 左に cherry。held は strawberry で左に落とすと埋める。
    # 右に strawberry があるので右で合成する。
    cherry_r = _radius(0)
    straw_r = _radius(1)
    floor_cherry = NORMALIZED_HEIGHT - cherry_r
    floor_straw = NORMALIZED_HEIGHT - straw_r
    buried = Fruit(type=0, x=100, y=floor_cherry, radius=cherry_r, confidence=90)
    mate = Fruit(type=1, x=300, y=floor_straw, radius=straw_r, confidence=90)
    x = choose_x(_obs(held_type=1, fruits=(buried, mate)))
    assert abs(x - mate.x) < straw_r * 3
    assert x > 200


def test_sets_up_next_when_no_immediate_merge() -> None:
    # held は grape。盤面に grape は無い。next は cherry で右に cherry がある。
    # 右寄りに置いて next の合成を用意する。
    cherry_r = _radius(0)
    grape_r = _radius(2)
    target = Fruit(type=0, x=300, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    # 左に大きな障害だけあると、低い右の next セットが勝つ。
    wall = Fruit(type=5, x=80, y=NORMALIZED_HEIGHT - _radius(5), radius=_radius(5), confidence=90)
    x = choose_x(_obs(held_type=2, fruits=(target, wall), next_type=0))
    assert x > NORMALIZED_WIDTH / 2
    # next cherry の近く (grape を隣に置く)。
    assert abs(x - target.x) < cherry_r + grape_r * 2 + 40


def test_small_fruit_goes_right_of_large() -> None:
    # 左に大きい実。小さい held は合成できないので右寄り (大きい順)。
    big_r = _radius(6)
    big = Fruit(type=6, x=90, y=NORMALIZED_HEIGHT - big_r, radius=big_r, confidence=90)
    x = choose_x(_obs(held_type=0, fruits=(big,)))
    assert x > NORMALIZED_WIDTH / 2


def test_prefers_held_that_enables_next_merge() -> None:
    # held=grape は盤に無く合成不可。右に cherry が1つ。next も cherry。
    # held を右の cherry 付近に置けば next が合成できる。左に置くと遠い。
    cherry_r = _radius(0)
    grape_r = _radius(2)
    cherry = Fruit(type=0, x=310, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    # 左にも床があるが、next 合成のために右を選ぶ。
    x = choose_x(_obs(held_type=2, fruits=(cherry,), next_type=0))
    assert x > NORMALIZED_WIDTH / 2
    assert abs(x - cherry.x) < cherry_r + grape_r * 2 + 50


def test_orange_stacks_on_left_apple() -> None:
    # 左端のリンゴ: 右隣の床が空いていれば、大きい順で右に並べる (上より隣)。
    apple_r = _radius(5)
    orange_r = _radius(4)
    apple = Fruit(type=5, x=apple_r + 8, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    side = apple.x + apple_r + orange_r
    x = choose_x(_obs(held_type=4, fruits=(apple,)))
    assert abs(x - side) < orange_r
    assert x > apple.x


def test_orange_stacks_on_apple_even_with_next_cherry() -> None:
    # next が右のサクランボでも、オレンジはリンゴの右隣へ。
    apple_r = _radius(5)
    orange_r = _radius(4)
    cherry_r = _radius(0)
    apple = Fruit(type=5, x=apple_r + 8, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    cherry = Fruit(type=0, x=300, y=NORMALIZED_HEIGHT - cherry_r, radius=cherry_r, confidence=90)
    side = apple.x + apple_r + orange_r
    x = choose_x(_obs(held_type=4, fruits=(apple, cherry), next_type=0))
    assert abs(x - side) < orange_r * 1.5
    assert x > apple.x
    assert x < NORMALIZED_WIDTH / 2


def test_orange_on_top_when_ordered_side_blocked() -> None:
    # リンゴの右隣が塞がっているときは上に積む。
    apple_r = _radius(5)
    orange_r = _radius(4)
    grape_r = _radius(2)
    apple = Fruit(type=5, x=apple_r + 8, y=NORMALIZED_HEIGHT - apple_r, radius=apple_r, confidence=90)
    side_x = apple.x + apple_r + orange_r
    blocker = Fruit(type=2, x=side_x, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    x = choose_x(_obs(held_type=4, fruits=(apple, blocker)))
    assert abs(x - apple.x) < apple_r * 0.9


def test_prefers_merging_wedged_small_over_stacking_on_larger() -> None:
    # リンゴ2つの間にブドウが挟まっている。held もブドウ。
    # 右にオレンジの並ぶ側が空いていても、挟まった同種の合成を優先する。
    apple_r = _radius(5)
    grape_r = _radius(2)
    orange_r = _radius(4)
    floor_apple = NORMALIZED_HEIGHT - apple_r
    floor_grape = NORMALIZED_HEIGHT - grape_r
    floor_orange = NORMALIZED_HEIGHT - orange_r
    left = Fruit(type=5, x=apple_r + 10, y=floor_apple, radius=apple_r, confidence=90)
    mid_x = left.x + apple_r + grape_r - 2
    wedged = Fruit(type=2, x=mid_x, y=floor_grape, radius=grape_r, confidence=90)
    right = Fruit(type=5, x=mid_x + grape_r + apple_r - 2, y=floor_apple, radius=apple_r, confidence=90)
    orange = Fruit(
        type=4,
        x=min(NORMALIZED_WIDTH - orange_r - 5, right.x + apple_r + orange_r + 30),
        y=floor_orange,
        radius=orange_r,
        confidence=90,
    )
    x = choose_x(_obs(held_type=2, fruits=(left, wedged, right, orange)))
    assert abs(x - wedged.x) < grape_r * 2
    after, merges = _after_drop(_obs(held_type=2, fruits=(left, wedged, right, orange)), x)
    assert merges >= 1


def test_stacks_strawberry_on_grape_when_next_is_strawberry() -> None:
    # NOTES: グレープを育てたいとき、held/next がイチゴならグレープの上へ。
    # 隣の床が空いていても、同種 next で育成する対象の上を選ぶ。
    grape_r = _radius(2)
    straw_r = _radius(1)
    grape = Fruit(type=2, x=120, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    x = choose_x(_obs(held_type=1, fruits=(grape,), next_type=1))
    assert abs(x - grape.x) < grape_r * 0.85


def test_grows_grape_even_with_distant_strawberry_merge() -> None:
    # 右に即合成できるイチゴがあっても、左のグレープ育成を優先する。
    grape_r = _radius(2)
    straw_r = _radius(1)
    grape = Fruit(type=2, x=100, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    lone = Fruit(type=1, x=320, y=NORMALIZED_HEIGHT - straw_r, radius=straw_r, confidence=90)
    x = choose_x(_obs(held_type=1, fruits=(grape, lone), next_type=1))
    assert abs(x - grape.x) < grape_r * 0.85
    assert abs(x - lone.x) > straw_r * 3


def test_stacks_grape_on_dekopon_when_next_is_grape() -> None:
    # 同じパターンの一段上: held/next がグレープならデコポンの上。
    dek_r = _radius(3)
    grape_r = _radius(2)
    dek = Fruit(type=3, x=110, y=NORMALIZED_HEIGHT - dek_r, radius=dek_r, confidence=90)
    x = choose_x(_obs(held_type=2, fruits=(dek,), next_type=2))
    assert abs(x - dek.x) < dek_r * 0.85


def test_orange_leaves_room_for_dekopon_beside_strawberry() -> None:
    # 右にチェリー・イチゴ。オレンジをイチゴのすぐ左へ置くと、
    # デコポン・グレープの並ぶ列が無くなるので、中間段階分だけ離す。
    cherry_r = _radius(0)
    straw_r = _radius(1)
    orange_r = _radius(4)
    cherry = Fruit(
        type=0,
        x=NORMALIZED_WIDTH - cherry_r - 2,
        y=NORMALIZED_HEIGHT - cherry_r,
        radius=cherry_r,
        confidence=90,
    )
    # ideal オレンジより左に寄ったイチゴ: 旧方策だとすぐ隣に置きがち。
    straw = Fruit(type=1, x=280, y=NORMALIZED_HEIGHT - straw_r, radius=straw_r, confidence=90)
    x = choose_x(_obs(held_type=4, fruits=(cherry, straw)))
    need = _chain_center_gap(4, 1)
    assert straw.x - x >= need - 4
    # 接触ぎりぎりよりは明らかに離す。
    assert straw.x - x - straw_r - orange_r > _radius(2) + _radius(3)


def test_grows_toward_large_fruit_on_the_right() -> None:
    # 大きい実が右に寄ったあとも左を大きくし直さない。リンゴは右のナシの左隣へ。
    pear_r = _radius(6)
    apple_r = _radius(5)
    pear = Fruit(type=6, x=300, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    side = pear.x - pear_r - apple_r
    x = choose_x(_obs(held_type=5, fruits=(pear,)))
    assert abs(x - side) < apple_r
    assert x < pear.x


def test_drop_on_slope_rolls_to_floor() -> None:
    # 大きい実の右肩に落とすと、側面を転がって床へ着地する。
    pear_r = _radius(6)
    cherry_r = _radius(0)
    pear = Fruit(type=6, x=200, y=NORMALIZED_HEIGHT - pear_r, radius=pear_r, confidence=90)
    drop_x = pear.x + pear_r * 0.55
    land_x, land_y = _preview_land((pear,), 0, drop_x, cherry_r)
    assert land_x > drop_x
    assert land_y >= NORMALIZED_HEIGHT - cherry_r - 1.0
    assert land_x >= pear.x + pear_r + cherry_r - 2.0


def test_merge_result_settles_from_midpoint() -> None:
    # 合成実は中点に出てから着地する (くっつき方向への寄り＋転がり)。
    cherry_r = _radius(0)
    floor_y = NORMALIZED_HEIGHT - cherry_r
    a = Fruit(type=0, x=200, y=floor_y, radius=cherry_r, confidence=90)
    b = Fruit(type=0, x=200 + 2 * cherry_r + 4, y=floor_y, radius=cherry_r, confidence=90)
    mid = (a.x + b.x) / 2
    after, merges = _after_drop(_obs(held_type=0, fruits=(a, b)), mid)
    assert merges >= 1
    grown = [f for f in after if f.type == 1]
    assert grown
    assert abs(grown[0].x - mid) < cherry_r * 2


def test_strawberry_does_not_roll_left_of_grape() -> None:
    # 序盤: ブドウ左上に置くと左へ転がって大小が崩れる。右隣か真上を選ぶ。
    grape_r = _radius(2)
    straw_r = _radius(1)
    grape = Fruit(type=2, x=160, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    obs = _obs(held_type=1, fruits=(grape,))
    x = choose_x(obs)
    land_x, _land = _preview_land((grape,), 1, x, straw_r)
    assert land_x >= grape.x - 1.0
    # 左肩落下は転がってブドウ左の床へ落ちるので選ばない。
    left_shoulder = grape.x - grape_r * 0.5
    assert _score(obs, left_shoulder, straw_r) < _score(obs, x, straw_r)


def test_left_shoulder_of_grape_rolls_to_left_floor() -> None:
    # シミュレーション自体が「左肩 → 左へ転がり」を再現すること。
    grape_r = _radius(2)
    straw_r = _radius(1)
    grape = Fruit(type=2, x=160, y=NORMALIZED_HEIGHT - grape_r, radius=grape_r, confidence=90)
    drop_x = grape.x - grape_r * 0.5
    land_x, land_y = _preview_land((grape,), 1, drop_x, straw_r)
    assert land_x < grape.x - grape_r
    assert land_y >= NORMALIZED_HEIGHT - straw_r - 1.0


def test_grape_stays_beside_right_edge_strawberry() -> None:
    # 1手目イチゴ右端、2手目ブドウは左隣に安定して着地する。
    # 接触で左端まで弾かれる手は選ばない (その後デコポンが置けず崩壊する筋)。
    straw_r = _radius(1)
    grape_r = _radius(2)
    straw = Fruit(
        type=1,
        x=NORMALIZED_WIDTH - straw_r - 2,
        y=NORMALIZED_HEIGHT - straw_r,
        radius=straw_r,
        confidence=90,
    )
    obs = _obs(held_type=2, fruits=(straw,))
    x = choose_x(obs)
    land_x, land_y = _preview_land((straw,), 2, x, grape_r)
    assert land_y >= NORMALIZED_HEIGHT - grape_r - 1.0
    assert land_x > NORMALIZED_WIDTH * 0.5
    assert abs(land_x - (straw.x - straw_r - grape_r - 4.0)) < grape_r
    # 肩に当てて左端へ滑る落としより、隙間付きの隣の方が良い。
    shoulder = straw.x - straw_r * 0.4
    assert _score(obs, x, grape_r) > _score(obs, shoulder, grape_r)


def test_pushes_near_orange_pair_from_outside() -> None:
    # 近いオレンジ2つ: 上に積むより、左外側から押してくっつける。
    # 右に大きい実があると右外側押しは不利 (doko2 と同型)。
    orange_r = _radius(4)
    apple_r = _radius(5)
    pear_r = _radius(6)
    floor_o = NORMALIZED_HEIGHT - orange_r
    left = Fruit(type=4, x=70, y=floor_o, radius=orange_r, confidence=90)
    right = Fruit(type=4, x=70 + 2 * orange_r + 20, y=floor_o, radius=orange_r, confidence=90)
    apple = Fruit(
        type=5,
        x=right.x + orange_r + apple_r + 10,
        y=NORMALIZED_HEIGHT - apple_r,
        radius=apple_r,
        confidence=90,
    )
    pear = Fruit(
        type=6,
        x=apple.x + apple_r + pear_r + 10,
        y=NORMALIZED_HEIGHT - pear_r,
        radius=pear_r,
        confidence=90,
    )
    obs = _obs(held_type=3, fruits=(left, right, apple, pear), next_type=4)
    x = choose_x(obs)
    assert x < left.x
    assert x <= left.x - left.radius * 0.45
