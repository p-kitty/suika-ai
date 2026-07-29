"""Drop just one move. For checking the learning loop works.

    python scripts/play_step.py           # dry-run (coordinates only)
    python scripts/play_step.py --live    # actually click
    python scripts/play_step.py --live 200  # specify the column
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env import Env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="actually click")
    parser.add_argument(
        "x",
        nargs="?",
        type=float,
        default=None,
        help="the drop column (normalized coordinates 0-400). The current held column when omitted",
    )
    args = parser.parse_args()

    env = Env(dry_run=not args.live)
    obs = env.reset()

    print(
        f"ready={obs.ready} blocked={obs.blocked} "
        f"held={obs.held_name}@{obs.held_x} next={obs.next_name} "
        f"fruits={len(obs.fruits)} dry_run={env.dry_run}"
    )
    if not obs.ready:
        print("not in a state to drop")
        return 1

    target = obs.held_x if args.x is None else args.x
    result = env.step(target)
    after = result.observation

    print(
        f"step info={result.info} target={result.target_x} done={result.done} "
        f"-> held={after.held_name}@{after.held_x} next={after.next_name} "
        f"fruits={len(after.fruits)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
