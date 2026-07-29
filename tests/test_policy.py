"""方策の単体テスト。画面は使わない。"""

from src.observe import Observation
from src.policy import _after_drop, _ideal_x, _land_y, _radius, choose_x
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
