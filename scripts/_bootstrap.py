"""The repository root shared by scripts.

By the time each script can import this module, ROOT is already on sys.path
(the sys.path.insert at the top when run directly, or the `python -m`
mechanism itself). So it has no function that adds the path.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
