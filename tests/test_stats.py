"""ペア比較の統計の単体テスト。"""

import math

from src.stats import (
    Z95,
    correlation,
    detect_n,
    paired_stats,
    pairing_gain,
    t_critical_95,
)


def test_t_critical_matches_table_and_approaches_normal() -> None:
    assert t_critical_95(1) == 12.706
    assert t_critical_95(30) == 2.042
    # 表の外は単調に減って正規の 1.96 へ寄る。
    assert 2.042 > t_critical_95(60) > t_critical_95(1000) > Z95


def test_constant_shift_is_significant() -> None:
    a = [100.0, 200.0, 300.0, 400.0, 500.0]
    b = [x + 50.0 for x in a]
    stats = paired_stats(a, b)
    assert stats.n == 5
    assert stats.delta == 50.0
    # ペアにすると個体差が消える。素の標準偏差は 100 超なのに差の分散は 0。
    assert stats.sd_diff == 0.0
    assert stats.sd_pooled > 100.0
    assert stats.ci_lo == 0.0 and stats.ci_hi == 0.0


def test_noise_swamps_small_effect() -> None:
    a = [1000.0, 3000.0, 2000.0, 500.0, 2500.0]
    b = [1200.0, 2600.0, 2400.0, 300.0, 2400.0]
    stats = paired_stats(a, b)
    assert stats.ci_lo < 0.0 < stats.ci_hi
    assert not stats.significant
    # ±100 を主張するには、この分散だと桁違いのエピソード数が要る。
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


def test_detect_n_is_unit_free() -> None:
    """指標の単位を変えても必要エピソード数は変わらない（だから比較に使える）。"""
    assert detect_n(50.0, 200.0) == detect_n(0.05, 0.2)
    # 差が同じならノイズが大きい指標ほど多く要る。
    assert detect_n(50.0, 400.0) > detect_n(50.0, 200.0)
    # 差が観測されていない指標は「何局要るか」を語れない。
    assert math.isnan(detect_n(0.0, 10.0))


def test_pairing_gain_baseline_is_independent_seeds() -> None:
    # 無相関なら差の SD は個体差の sqrt(2) 倍。そこが利得ゼロの基準。
    assert pairing_gain(math.sqrt(2.0) * 10.0, 10.0) == 0.0
    # 差の分散が消えるほど 1 に近づく。
    assert pairing_gain(0.0, 10.0) == 1.0
    # 対にすると却って荒れる場合は負。
    assert pairing_gain(30.0, 10.0) < 0.0
    assert math.isnan(pairing_gain(1.0, 0.0))


def test_correlation_handles_constant_series() -> None:
    assert correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    assert math.isnan(correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
