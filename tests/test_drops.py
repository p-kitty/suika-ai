"""Read screenshots/doko*.png and have it solve where to place.

localize → Observation → choose_x.

held/next compare the detection against the ground truth in expected_drops (not assuming expected).
expect_x is the allowed range of the drop column.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from src.imagefile import read
from src.observe import from_board
from src.policy import choose_x
from src.vision.board import localize
from tests.expected_drops import EXPECTED_DROPS, KNOWN_DROP_FAILURES

SCREENSHOTS = Path(__file__).resolve().parents[1] / "screenshots"


@lru_cache(maxsize=None)
def _localize(name: str):
    image = read(SCREENSHOTS / name)
    return None if image is None else localize(image)


def _cases() -> list:
    return [
        pytest.param(
            name,
            marks=[pytest.mark.xfail(reason=KNOWN_DROP_FAILURES[name], strict=True)]
            if name in KNOWN_DROP_FAILURES
            else [],
        )
        for name in sorted(
            EXPECTED_DROPS,
            key=lambda n: int(Path(n).stem.replace("doko", "")),
        )
    ]


@pytest.mark.parametrize("name", _cases())
def test_drop_from_screenshot(name: str) -> None:
    result = _localize(name)
    if result is None:
        pytest.skip(f"screenshots/{name} is missing")

    case = EXPECTED_DROPS[name]
    obs = from_board(result)
    assert obs.ready, f"{name}: cannot read board/held (blocked={obs.blocked})"
    assert obs.held_name == case["held"], (
        f"held detected={obs.held_name} expected={case['held']}"
    )
    assert obs.next_name == case.get("next"), (
        f"next detected={obs.next_name} expected={case.get('next')}"
    )
    assert obs.held_type is not None

    x = choose_x(obs)
    lo, hi = case["expect_x"]
    assert lo <= x <= hi, f"expect_x [{lo}, {hi}]: x={x:.1f}"
