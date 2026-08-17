"""保存済みのスクリーンショットを正解と突き合わせる。

しきい値をいじると別のシーンが壊れる。目で見て起こした正解を置いておいて、
何を直したのか・何を壊したのかがそのまま出るようにする。

正解データなので screenshots/ は git で追跡する。それでも画像がなければ
その画像ぶんは skip する。正解だけ先に書き足せるようにしてある。
"""

import math
from functools import lru_cache
from pathlib import Path

import pytest

from src.util.imagefile import read
from src.vision.board import localize
from src.vision.classify import fruit_radius_ratios
from src.vision.colors import FRUIT_NAMES
from src.vision.normalized import NORMALIZED_WIDTH
from src.vision.state import Fruit
from tests.vision.expected_fruits import (
    BLOCKED,
    EXPECTED,
    EXPECTED_HELD,
    EXPECTED_NEXT,
    KNOWN_FAILURES,
)

SCREENSHOTS = Path(__file__).resolve().parents[2] / "screenshots"


@lru_cache(maxsize=None)
def _localize(name: str):
    """1 枚につき一度だけ検出する。localize は数十 ms かかる。"""
    image = read(SCREENSHOTS / name)
    return None if image is None else localize(image)


def _result(name: str):
    result = _localize(name)
    if result is None:
        pytest.skip(f"screenshots/{name} がない")
    return result


def _ordered(names) -> list[str]:
    return sorted(names, key=lambda name: int(Path(name).stem))


def _fruit_cases() -> list:
    """既知の取り違えには xfail を付ける。

    strict なので直ったら失敗する。そのとき KNOWN_FAILURES から外す。
    """
    return [
        pytest.param(
            name,
            marks=[pytest.mark.xfail(reason=KNOWN_FAILURES[name], strict=True)]
            if name in KNOWN_FAILURES
            else [],
        )
        for name in _ordered(EXPECTED)
    ]


def _pair_up(expected: list[tuple], detected: list[Fruit]) -> dict[int, int]:
    """正解と検出を位置で一対一に対応づける。

    近い組から順に確定させる。許容はそのフルーツの半径ぶんで、中心が円の
    内側を指していれば同じフルーツを指したものとみなす。型は見ないので、
    段階を読み違えただけの検出は「誤分類」として残り、取り逃がしにならない。
    """
    ratios = fruit_radius_ratios()
    candidates = []

    for want, (name, x, y) in enumerate(expected):
        limit = ratios[FRUIT_NAMES.index(name)] * NORMALIZED_WIDTH
        for got, fruit in enumerate(detected):
            distance = math.hypot(fruit.x - x, fruit.y - y)
            if distance <= limit:
                candidates.append((distance, want, got))

    pairs: dict[int, int] = {}
    taken: set[int] = set()

    for _, want, got in sorted(candidates):
        if want not in pairs and got not in taken:
            pairs[want] = got
            taken.add(got)

    return pairs


def _differences(expected: list[tuple], detected: list[Fruit]) -> list[str]:
    pairs = _pair_up(expected, detected)
    matched = set(pairs.values())
    lines = []

    for want, (name, x, y) in enumerate(expected):
        if want not in pairs:
            lines.append(f"取り逃がし {name:11s} ({x:3d},{y:3d})")
            continue

        fruit = detected[pairs[want]]
        if fruit.name != name:
            lines.append(
                f"誤分類     {name:11s} ({x:3d},{y:3d}) -> "
                f"{fruit.name} ({fruit.x:3.0f},{fruit.y:3.0f}) r={fruit.radius:.1f}"
            )

    for got, fruit in enumerate(detected):
        if got not in matched:
            lines.append(
                f"余分       {fruit.name:11s} ({fruit.x:3.0f},{fruit.y:3.0f}) "
                f"r={fruit.radius:.1f} 信頼度 {fruit.confidence:.0f}%"
            )

    return lines


@pytest.mark.parametrize("name", _ordered(list(EXPECTED) + list(BLOCKED)))
def test_board_found(name: str) -> None:
    assert _result(name).found


@pytest.mark.parametrize("name", _ordered(BLOCKED))
def test_dialog_hides_board(name: str) -> None:
    result = _result(name)

    assert result.blocked
    # 覆われている間は読めない。古い盤面を返さないこと。
    assert result.fruits is None
    assert result.held_fruit is None
    assert result.next_fruit is None


@pytest.mark.parametrize("name", _ordered(EXPECTED_HELD))
def test_held_fruit(name: str) -> None:
    """雲が持っている、次に落ちるフルーツ。"""
    result = _result(name)
    held = result.held_fruit
    expected_name, expected_x = EXPECTED_HELD[name]

    assert (
        held is not None
        and held.fruit is not None
        and held.x is not None
        and held.y is not None
        and held.radius is not None
    ), f"{name}: 落下待ちフルーツを取り逃がした"

    detail = f"{held.fruit.name} x={held.x:.0f} r={held.radius:.1f} 上辺から {-held.y:.0f}"
    assert held.fruit.name == expected_name, f"{name}: 誤分類 {expected_name} -> {detail}"
    # 落とす列。半径ぶん以上ずれていたら別の塊を拾っている。
    assert abs(held.x - expected_x) <= held.radius, f"{name}: 位置ちがい {expected_x} -> {detail}"


@pytest.mark.parametrize("name", _ordered(EXPECTED_NEXT))
def test_next_fruit(name: str) -> None:
    """next の泡の中身。落下待ちのさらに次に来るフルーツ。"""
    result = _result(name)
    next_fruit = result.next_fruit

    assert next_fruit is not None and next_fruit.fruit is not None, (
        f"{name}: next のフルーツを取り逃がした"
    )

    detail = f"{next_fruit.fruit.name} 比 {next_fruit.radius_ratio:.3f}"
    assert next_fruit.fruit.name == EXPECTED_NEXT[name], (
        f"{name}: 誤分類 {EXPECTED_NEXT[name]} -> {detail}"
    )


@pytest.mark.parametrize("name", _fruit_cases())
def test_fruits(name: str) -> None:
    result = _result(name)
    assert not result.blocked

    detected = result.fruits or []
    expected = EXPECTED[name]
    differences = _differences(expected, detected)

    assert not differences, "\n".join(
        [f"{name}: 検出 {len(detected)} 個 / 正解 {len(expected)} 個", *differences]
    )
