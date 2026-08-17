"""Small tools for testing A/B differences paired on the same seeds.

Score noise is large, and looking only at a rise or fall in the mean always ends with 'it feels better'.
Here the statistic is the **difference** from running the same seeds through both variants (a paired comparison).
Per-seed variation cancels out, so it is far more sensitive than comparing raw means.

scipy is not installed, so the t critical values are a lookup table plus a large-sample approximation.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# Two-sided 95% t critical values. Table lookup for df=1..30, approaching the normal approximation beyond that.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
# The limit as df→∞ (the 97.5% point of the standard normal).
Z95 = 1.960


def t_critical_95(df: int) -> float:
    """Two-sided 95% t critical value for df degrees of freedom."""
    if df < 1:
        return float("nan")
    if df in _T95:
        return _T95[df]
    # For df>30 it is nearly linear in 1/df. Interpolate between the df=30 value and the limit.
    return Z95 + (_T95[30] - Z95) * (30.0 / df)


@dataclass(frozen=True)
class PairedStats:
    """Statistics of the A/B difference paired on the same seeds."""

    n: int
    mean_a: float
    mean_b: float
    sd_diff: float  # SD of the difference (variance of the paired comparison)
    sd_pooled: float  # SD of per-game variation (a guide for the unpaired case)
    t: float
    ci_lo: float
    ci_hi: float

    @property
    def delta(self) -> float:
        return self.mean_b - self.mean_a

    @property
    def se(self) -> float:
        """Standard error of the mean difference."""
        return self.sd_diff / math.sqrt(self.n) if self.n > 1 else float("nan")

    @property
    def significant(self) -> bool:
        """Whether the 95% CI does not cross 0."""
        if self.ci_lo != self.ci_lo:  # NaN
            return False
        return self.ci_lo > 0.0 or self.ci_hi < 0.0

    def required_n(self, target: float) -> float:
        """Episodes needed to bring the 95% CI half-width within ±target.

        Not the n needed to **detect** that difference significantly (accounting for power
        needs nearly twice as many). Only a lower-bound guide: 'to make a claim of target width with this metric,
        run at least this many'.
        """
        if target <= 0 or self.sd_diff != self.sd_diff:
            return float("nan")
        return (Z95 * self.sd_diff / target) ** 2


def paired_stats(a: list[float], b: list[float]) -> PairedStats:
    """Compute difference statistics from two paired samples. Pairs containing NaN are dropped."""
    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y]
    n = len(pairs)
    nan = float("nan")
    if n == 0:
        return PairedStats(0, nan, nan, nan, nan, nan, nan, nan)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    diffs = [y - x for x, y in pairs]
    mean_a = statistics.mean(xs)
    mean_b = statistics.mean(ys)
    if n < 2:
        return PairedStats(n, mean_a, mean_b, nan, nan, nan, nan, nan)
    sd_diff = statistics.stdev(diffs)
    sd_pooled = statistics.stdev(xs + ys)
    mean_diff = statistics.mean(diffs)
    if sd_diff == 0.0:
        # Every pair is identical. The difference is exactly 0, so t is left undefined.
        return PairedStats(n, mean_a, mean_b, 0.0, sd_pooled, nan, 0.0, 0.0)
    se = sd_diff / math.sqrt(n)
    half = t_critical_95(n - 1) * se
    return PairedStats(
        n=n,
        mean_a=mean_a,
        mean_b=mean_b,
        sd_diff=sd_diff,
        sd_pooled=sd_pooled,
        t=mean_diff / se,
        ci_lo=mean_diff - half,
        ci_hi=mean_diff + half,
    )


def detect_n(delta: float, sd_diff: float) -> float:
    """Episodes needed to move an observed difference delta away from 0 at the 95% CI.

    A yardstick for choosing metrics. It lines up as 'how many games to see this change with this metric' regardless of units,
    so metrics in different units such as score and moves survived can be compared.
    """
    if delta == 0.0 or delta != delta or sd_diff != sd_diff:
        return float("nan")
    if sd_diff == 0.0:
        return 2.0  # zero variance. Two pairs are enough.
    return (Z95 * sd_diff / abs(delta)) ** 2


def pairing_gain(sd_diff: float, sd_pooled: float) -> float:
    """The fraction by which pairing on the same seeds shrank the SE of the difference.

    Measuring an A/B on independent seeds makes the SD of the difference sqrt(2) times the per-game variation. That is
    taken as the 'unpaired' reference, so 0 means no gain, and negative means A and B are
    anti-correlated and pairing actually makes it noisier.
    """
    if sd_pooled != sd_pooled or sd_pooled == 0.0 or sd_diff != sd_diff:
        return float("nan")
    return 1.0 - sd_diff / (math.sqrt(2.0) * sd_pooled)


def correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation after dropping NaN. NaN for a constant series."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 2:
        return float("nan")
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    try:
        return statistics.correlation(a, b)
    except statistics.StatisticsError:
        # One side is constant (every episode dead and so on). The correlation is undefined.
        return float("nan")
