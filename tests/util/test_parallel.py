"""`--workers` resolution. The scripts pass their raw CLI value, so both sentinels have to work."""

from __future__ import annotations

import os

from src.util.parallel import default_workers, resolve_workers


def test_both_sentinels_mean_auto() -> None:
    """None (compare_policy, eval_policy) and 0 (collect_value, train_sim) are the same request."""
    assert resolve_workers(None) == default_workers()
    assert resolve_workers(0) == default_workers()


def test_an_explicit_count_is_kept() -> None:
    assert resolve_workers(3) == 3
    assert resolve_workers(1) == 1


def test_a_negative_count_falls_back_to_serial() -> None:
    """Never hand a ProcessPoolExecutor a count below 1; it raises."""
    assert resolve_workers(-4) == 1


def test_auto_leaves_cores_for_the_real_game() -> None:
    """The real game runs on the same machine and stutters once every core is busy."""
    cores = os.cpu_count() or 4
    assert 1 <= default_workers() < cores
