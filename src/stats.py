"""同一シードでペアにした A/B の差を検定するための小道具。

score のノイズが大きく、平均の増減だけを見ると毎回「良くなった気がする」で
終わる。ここでは同じシードを両変種に流した**差**を統計量にする（対応のある比較）。
シードごとの個体差が差し引かれるぶん、素の平均比較よりずっと感度が高い。

scipy は入れていないので t 分布の臨界値は表引き + 大標本近似で済ませる。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# 両側 95% の t 臨界値。df=1..30 まで表引きし、それ以上は正規近似に寄せる。
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
# df→∞ の極限（標準正規の 97.5% 点）。
Z95 = 1.960


def t_critical_95(df: int) -> float:
    """自由度 df に対する両側 95% の t 臨界値。"""
    if df < 1:
        return float("nan")
    if df in _T95:
        return _T95[df]
    # df>30 は 1/df に対してほぼ線形。df=30 の値と極限を内挿する。
    return Z95 + (_T95[30] - Z95) * (30.0 / df)


@dataclass(frozen=True)
class PairedStats:
    """同一シードで対にした A/B の差の統計。"""

    n: int
    mean_a: float
    mean_b: float
    sd_diff: float  # 差の標準偏差（ペア比較の分散）
    sd_pooled: float  # 個体差の標準偏差（ペアにしなかった場合の目安）
    t: float
    ci_lo: float
    ci_hi: float

    @property
    def delta(self) -> float:
        return self.mean_b - self.mean_a

    @property
    def se(self) -> float:
        """差の平均の標準誤差。"""
        return self.sd_diff / math.sqrt(self.n) if self.n > 1 else float("nan")

    @property
    def significant(self) -> bool:
        """95% CI が 0 をまたがないか。"""
        if self.ci_lo != self.ci_lo:  # NaN
            return False
        return self.ci_lo > 0.0 or self.ci_hi < 0.0

    def required_n(self, target: float) -> float:
        """±target を 95% CI の半幅に収めるのに必要なエピソード数。

        「その差を有意に**検出**する」のに要る n ではない（検出力を考えると
        さらに倍近く要る）。あくまで「この指標で target 幅の主張をするなら
        最低これだけ回す」という下限の目安。
        """
        if target <= 0 or self.sd_diff != self.sd_diff:
            return float("nan")
        return (Z95 * self.sd_diff / target) ** 2


def paired_stats(a: list[float], b: list[float]) -> PairedStats:
    """対応のある 2 標本から差の統計を出す。NaN を含むペアは捨てる。"""
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
        # 全ペアで完全に同じ。差は 0 で確定なので t は定義しない。
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


def correlation(xs: list[float], ys: list[float]) -> float:
    """NaN を落としたうえでの Pearson 相関。定数列なら NaN。"""
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 2:
        return float("nan")
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    try:
        return statistics.correlation(a, b)
    except statistics.StatisticsError:
        # どちらかが定数（全エピソード dead など）。相関は定義できない。
        return float("nan")
