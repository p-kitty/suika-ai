"""Shared fixtures. Tests pin move selection with an exact aim.

`policy.AIM_SPREAD` scores a held candidate over the columns the real game may release at
(NOTES 'Measured: aim error costs a fifth of the score'). That is right for play, but it blurs every
position built to pin one decision: a gap one dekopon wide also scores the drops that close it.
So the whole suite runs with the spread off, and only the tests about the spread itself turn it back on.

`DEFAULT_AIM_SPREAD` keeps what the policy ships with, since the fixture hides the module value.
"""

from __future__ import annotations

import pytest

from src import policy as pol

DEFAULT_AIM_SPREAD = pol.AIM_SPREAD


@pytest.fixture(autouse=True)
def exact_aim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pol, "AIM_SPREAD", 0.0)
