"""Unit tests for paired comparison statistics."""

import math

from src.stats import Z95, correlation, paired_stats, t_critical_95


def test_t_critical_matches_table_and_approaches_normal() -> None:
    assert t_critical_95(1) == 12.706
    assert t_critical_95(30) == 2.042
    # Beyond the table it decreases monotonically toward the normal 1.96.
    assert 2.042 > t_critical_95(60) > t_critical_95(1000) > Z95


def test_constant_shift_is_significant() -> None:
    a = [100.0, 200.0, 300.0, 400.0, 500.0]
    b = [x + 50.0 for x in a]
    stats = paired_stats(a, b)
    assert stats.n == 5
    assert stats.delta == 50.0
    # Pairing removes per-game variation. The raw SD is over 100 yet the variance of the difference is 0.
    assert stats.sd_diff == 0.0
    assert stats.sd_pooled > 100.0
    assert stats.ci_lo == 0.0 and stats.ci_hi == 0.0


def test_noise_swamps_small_effect() -> None:
    a = [1000.0, 3000.0, 2000.0, 500.0, 2500.0]
    b = [1200.0, 2600.0, 2400.0, 300.0, 2400.0]
    stats = paired_stats(a, b)
    assert stats.ci_lo < 0.0 < stats.ci_hi
    assert not stats.significant
    # Claiming ±100 with this variance needs orders of magnitude more episodes.
    assert stats.required_n(100.0) > stats.n


def test_required_n_scales_with_variance() -> None:
    quiet = paired_stats([1.0, 2.0, 3.0, 4.0], [1.1, 2.0, 3.1, 4.0])
    noisy = paired_stats([1.0, 2.0, 3.0, 4.0], [3.0, 0.5, 5.0, 2.0])
    assert quiet.required_n(0.1) < noisy.required_n(0.1)


def test_nan_pairs_are_dropped() -> None:
    stats = paired_stats([1.0, float("nan"), 3.0], [2.0, 5.0, 4.0])
    assert stats.n == 2
    assert stats.mean_a == 2.0


def test_empty_and_single_pair_are_not_errors() -> None:
    assert paired_stats([], []).n == 0
    single = paired_stats([1.0], [2.0])
    assert single.n == 1
    assert math.isnan(single.sd_diff)
    assert not single.significant


def test_correlation_handles_constant_series() -> None:
    assert correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    assert math.isnan(correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
