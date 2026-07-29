"""Compare saved screenshots against the ground truth.

Touching thresholds breaks other scenes. Keeping ground truth transcribed by eye
makes what was fixed and what was broken show up directly.

It is ground truth, so screenshots/ is tracked in git. Even so, if an image is missing
that image is skipped. This allows adding ground truth first.
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
from tests.expected_fruits import BLOCKED, EXPECTED, EXPECTED_HELD, KNOWN_FAILURES

SCREENSHOTS = Path(__file__).resolve().parents[1] / "screenshots"


@lru_cache(maxsize=None)
def _localize(name: str):
    """Detect only once per image. localize takes tens of ms."""
    image = read(SCREENSHOTS / name)
    return None if image is None else localize(image)


def _result(name: str):
    result = _localize(name)
    if result is None:
        pytest.skip(f"screenshots/{name} is missing")
    return result


def _ordered(names) -> list[str]:
    return sorted(names, key=lambda name: int(Path(name).stem))


def _fruit_cases() -> list:
    """Mark known misreads with xfail.

    It is strict, so it fails once fixed. Remove it from KNOWN_FAILURES then.
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
    """Match ground truth and detections one to one by position.

    Pairs are fixed from the closest. The tolerance is that fruit's radius: if the center points
    inside the circle it is considered the same fruit. Types are not looked at, so
    a detection that only misread the stage remains as a 'misclassification', not a miss.
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
            lines.append(f"missed         {name:11s} ({x:3d},{y:3d})")
            continue

        fruit = detected[pairs[want]]
        if fruit.name != name:
            lines.append(
                f"misclassified  {name:11s} ({x:3d},{y:3d}) -> "
                f"{fruit.name} ({fruit.x:3.0f},{fruit.y:3.0f}) r={fruit.radius:.1f}"
            )

    for got, fruit in enumerate(detected):
        if got not in matched:
            lines.append(
                f"extra          {fruit.name:11s} ({fruit.x:3.0f},{fruit.y:3.0f}) "
                f"r={fruit.radius:.1f} confidence {fruit.confidence:.0f}%"
            )

    return lines


@pytest.mark.parametrize("name", _ordered(list(EXPECTED) + list(BLOCKED)))
def test_board_found(name: str) -> None:
    assert _result(name).found


@pytest.mark.parametrize("name", _ordered(BLOCKED))
def test_dialog_hides_board(name: str) -> None:
    result = _result(name)

    assert result.blocked
    # Cannot be read while covered. Must not return an old board.
    assert result.fruits is None
    assert result.held_fruit is None


@pytest.mark.parametrize("name", _ordered(EXPECTED_HELD))
def test_held_fruit(name: str) -> None:
    """The next fruit to fall, held by the cloud."""
    result = _result(name)
    held = result.held_fruit
    expected_name, expected_x = EXPECTED_HELD[name]

    assert held is not None and held.fruit is not None and held.x is not None, (
        f"{name}: missed the waiting fruit"
    )

    detail = f"{held.fruit.name} x={held.x:.0f} r={held.radius:.1f} above the top edge {-held.y:.0f}"
    assert held.fruit.name == expected_name, f"{name}: misclassified {expected_name} -> {detail}"
    # The drop column. Off by more than the radius means it picked up another blob.
    assert abs(held.x - expected_x) <= held.radius, f"{name}: wrong position {expected_x} -> {detail}"


@pytest.mark.parametrize("name", _fruit_cases())
def test_fruits(name: str) -> None:
    result = _result(name)
    assert not result.blocked

    detected = result.fruits or []
    expected = EXPECTED[name]
    differences = _differences(expected, detected)

    assert not differences, "\n".join(
        [f"{name}: detected {len(detected)} / expected {len(expected)}", *differences]
    )
