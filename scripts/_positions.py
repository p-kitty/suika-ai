"""Position sampling and the position cache, shared by the escape screens.

`band_escape.py`, `weight_escape.py` and `value_escape.py` all do the same thing before their own
analysis: play seeds with the current policy, keep one position every `--stride` moves from `--skip`
onward, turn each into a per-script row, and pickle the result so reruns are analytic.
Only the row differs, so only the row builder is per script.

Not a script. It has no sys.path insert; by the time a script imports it ROOT is already on the path
(see `_bootstrap.py`).
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypeVar

from src.observe import Observation
from src.policy import choose_x
from src.sim.sim_env import SimEnv

# A row of the table. Each script defines its own (a bare Observation, a breakdown, post-drop boards).
Row = TypeVar("Row")
# Observation, seed and move number -> one row, or None to drop the position.
Builder = Callable[[Observation, int, int], "Row | None"]


def add_sampling_args(
    parser: argparse.ArgumentParser, *, skip: int, stride: int, cache: Path, cache_help: str
) -> None:
    """The `--seeds`/`--seed`/`--steps`/`--skip`/`--stride`/`--eps`/`--positions` block.

    skip and stride have no shared default: the band is read late in the game, while terms divided by the
    fruit count have to be read from move 0. The caller passes what that screen looks at.
    """
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--skip", type=int, default=skip)
    parser.add_argument("--stride", type=int, default=stride)
    parser.add_argument("--eps", type=float, default=0.1, help="width of the tie band")
    parser.add_argument("--positions", type=Path, default=cache, help=cache_help)


def seeds_of(args: argparse.Namespace) -> list[int]:
    """`--seeds` consecutive seeds from `--seed`."""
    return [args.seed + i for i in range(args.seeds)]


def iter_positions(
    seed: int, steps: int, skip: int, stride: int
) -> Iterator[tuple[int, Observation]]:
    """Play one game with `choose_x` and yield (move number, position) at the sampled moves.

    The line played does not depend on the sampling, so changing skip or stride only changes which
    positions are kept, never the game.
    """
    env = SimEnv(seed=seed)
    obs = env.reset()
    for step in range(steps):
        if obs.held_type is None:
            break
        if step >= skip and step % stride == 0:
            yield step, obs
        result = env.step(choose_x(obs))
        obs = result.observation
        if result.done:
            break


def _collect_seed(job: tuple[int, int, int, int, Builder]) -> list[Row]:
    seed, steps, skip, stride, build = job
    rows: list[Row] = []
    for step, obs in iter_positions(seed, steps, skip, stride):
        row = build(obs, seed, step)
        if row is not None:
            rows.append(row)
    return rows


def collect(
    seeds: list[int],
    steps: int,
    skip: int,
    stride: int,
    build: Builder,
    *,
    workers: int = 1,
) -> list[Row]:
    """Rows of every sampled position over the seeds, in seed order.

    workers > 1 runs one process per seed. Seeds are independent games, so the table is the same as the
    serial one. `build` is sent to the workers by reference, so it has to be a module-level function.
    """
    jobs = [(seed, steps, skip, stride, build) for seed in seeds]
    table: list[Row] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            per_seed = list(pool.map(_collect_seed, jobs))
    else:
        per_seed = [_collect_seed(job) for job in jobs]
    for seed, rows in zip(seeds, per_seed):
        table.extend(rows)
        print(f"  seed {seed}: positions {len(rows)} (total {len(table)})", flush=True)
    return table


def load_or_build(
    path: Path,
    build: Callable[[], list[Row]],
    *,
    validate: Callable[[list[Row]], None] | None = None,
) -> list[Row]:
    """Reuse the pickled table if it is there, otherwise build it and save it.

    validate runs on a reused table. Raise from it when the cache predates a format change; a table
    read on the wrong assumption produces numbers that look fine.
    """
    if path.exists():
        table = pickle.loads(path.read_bytes())
        if validate is not None:
            validate(table)
        print(f"reusing cache {path}")
        return table
    table = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(table))
    print(f"saved cache: {path}")
    return table
