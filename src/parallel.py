import os


def default_workers() -> int:
    """CPU-bound 向け。論理コアの半分 (9700X なら 8)。"""
    n = os.cpu_count() or 4
    return max(1, n // 2)
