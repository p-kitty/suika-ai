import os


def default_workers() -> int:
    """For CPU-bound work. Half the logical cores (8 on a 9700X)."""
    n = os.cpu_count() or 4
    return max(1, n // 2)
