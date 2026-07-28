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

from src.imagefile import read
from src.vision.board import NORMALIZED_WIDTH, localize
from src.vision.classify import fruit_radius_ratios
from src.vision.colors import FRUIT_NAMES
from src.vision.state import Fruit
from tests.expected_fruits import BLOCKED, EXPECTED, KNOWN_FAILURES

SCREENSHOTS = Path(__file__).resolve().parents[1] / "screenshots"


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
