"""Process counts for CPU-bound scripts."""

from __future__ import annotations

import os


def default_workers() -> int:
    """For CPU-bound work. Half the logical cores (8 on a 9700X).

    Not all cores: the real game is played on the same machine, and it stutters once every core is busy.
    """
    n = os.cpu_count() or 4
    return max(1, n // 2)


def resolve_workers(requested: int | None) -> int:
    """A `--workers` value from the command line -> the process count. None and 0 both mean auto.

    Scripts differ in which of the two they use as the argparse default, so both are accepted here rather than
    each script spelling the fallback out (they once disagreed, and one had the count of this machine baked in).
    """
    if requested is None or requested == 0:
        return default_workers()
    return max(1, requested)
