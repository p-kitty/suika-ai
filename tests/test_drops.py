"""screenshots/doko*.png を読んで置き場所を解かせる。

localize → Observation → choose_x。

held/next は expected_drops の正解と検出を突き合わせる (expected 前提ではない)。
expect_x は落とす列の許容範囲。
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
        pytest.skip(f"screenshots/{name} がない")

    case = EXPECTED_DROPS[name]
    obs = from_board(result)
    assert obs.ready, f"{name}: 盤面/held が読めない (blocked={obs.blocked})"
    assert obs.held_name == case["held"], (
        f"held 検出={obs.held_name} 期待={case['held']}"
    )
    assert obs.next_name == case.get("next"), (
        f"next 検出={obs.next_name} 期待={case.get('next')}"
    )
    assert obs.held_type is not None

    x = choose_x(obs)
    lo, hi = case["expect_x"]
    assert lo <= x <= hi, f"expect_x [{lo}, {hi}]: x={x:.1f}"
