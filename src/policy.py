"""Decide the drop column. A thin bootstrap policy (the groundwork for RL).

It has no concrete procedures (push-ins, restoring pushes, cascade gap opening, ladder firing and the like).
It only looks at merging, burying, light size order and rolling accident prevention.
Moves are scored as eval = score (the real game's merge points) - penalties (penalties for accidents and bad moves).
Only dying moves are not compared by eval: they are removed from the candidates if a living move exists (`choose_x`).

The penalty side is `penalties.py`. This file only generates candidate columns, evaluates one move and looks ahead to next.
They are referenced as `pen.X` because the A/B in scripts/compare_policy.py
swaps weights by rewriting module attributes (do not bind them at import).
"""

from __future__ import annotations

import itertools
import math
from concurrent.futures import Executor

from . import penalties as pen
from .observe import Observation, clamp_drop_x
from .reward import is_lost, merge_score
from .sim.sim_physics import landed_xy
from .sim.sim_physics import simulate_drop_held
from .vision.classify import fruit_radius
from .vision.colors import SPAWN_MAX_TYPE
from .vision.normalized import NORMALIZED_WIDTH
from .vision.state import Fruit

# --- Lookahead and candidate coarseness ---
# Discount for the next move.
NEXT_DISCOUNT = 0.55
# Search coarseness. The physics (simulate_drop) dominates, and this nearly decides the run time.
# The idea of widening (8/16) was shelved at score +3.8% for 3.68x the cost
# (NOTES 'Run cost: faster physics and search width').
# Number of held candidates that get the next lookahead. The physics is heavy, so only the top.
HELD_TOP = 2
# Candidate spacing of the next lookahead. Coarser than held (CANDIDATE_STEP).
NEXT_CANDIDATE_STEP = 32.0
# Uniform spacing of held candidates. Coarser puts the spot directly above a dangerous pile among the candidates, so do not raise it
# (test_avoids_dangerous_tall_stack failed at 20). Speed is earned on the lookahead side.
# The finer side is closed too. The window for rolling into a same-type fruit can be only 1-3px,
# and lowering to 3.0 catches it, but score did not move against 248 → 540ms per move
# (NOTES 'Candidate spacing and the merge window').
CANDIDATE_STEP = 12.0


def _held_eval_job(
    obs: Observation, held_r: float, x: float
) -> tuple[float, float, list[Fruit], float]:
    """(eval, x, after, real-game score) for one held candidate. The unit sent to the pool.

    The real-game score is returned separately from eval because value-function training ranks candidates
    as Q = real-game score + V(after) (`training/collect.py`). eval is score - penalties, so
    it cannot be recovered from that.
    """
    after, held_eval, score = _held_eval(obs, x, held_r)
    return held_eval, x, after, score


def rank_candidates(
    obs: Observation, *, pool: Executor | None = None
) -> list[tuple[float, float, list[Fruit], float]]:
    """Return (eval, x, post-drop board, real-game score) per candidate in descending eval order.

    This is exactly the first-ply evaluation of `choose_x`. It is cut before the next lookahead so that
    training data collection can get the candidate table without running the same physics twice
    (`training/collect.py`). Nearly all of a move is `simulate_drop`, so
    recomputing it on the collection side would simply double the cost (NOTES 'Run cost: faster physics and search width').
    """
    if obs.held_type is None:
        raise ValueError("no held_type")

    held_r = fruit_radius(obs.held_type)
    xs = [
        clamp_drop_x(x, obs.held_type)
        for x in _candidates(list(obs.fruits), obs.held_type, held_r, extra_type=obs.next_type)
    ]
    if not xs:
        return []

    if pool is None:
        ranked = [_held_eval_job(obs, held_r, x) for x in xs]
    else:
        ranked = list(
            pool.map(_held_eval_job, itertools.repeat(obs), itertools.repeat(held_r), xs)
        )

    ranked.sort(key=lambda row: row[0], reverse=True)
    return ranked


def choose_x(
    obs: Observation,
    *,
    pool: Executor | None = None,
    ranked: list[tuple[float, float, list[Fruit], float]] | None = None,
) -> float:
    """Return the column to drop from the observation. Assumes ready with held_type present.

    Passing pool spreads the simulate_drop of held/next candidates over a process pool.
    The result is the same as serial execution (every candidate is independent and the board is only read).

    Passing the result of `rank_candidates` as ranked avoids recomputing the first-ply physics.
    Used by the collection side that wants the candidate table itself (`training/collect.py`) to have the teacher's move
    decided here again. Do not copy the move-selection rules into it.
    """
    if ranked is None:
        ranked = rank_candidates(obs, pool=pool)
    if not ranked:
        return NORMALIZED_WIDTH / 2

    # Dying moves are not compared by eval. Expressed as a penalty, the amount saved by avoiding a dirty board
    # outweighs the weight of death and it commits suicide (5 cases in 428 positions, chosen while 30-45 living moves existed).
    # The difference was up to 261, so no finite penalty is enough.
    alive = [row for row in ranked if not is_lost(row[2])]
    if alive:
        ranked = alive
    if obs.next_type is None:
        return ranked[0][1]

    # The next lookahead covers only the top held eval (the physics is heavy). Candidates are coarser than held.
    best_x = ranked[0][1]
    best_score = -math.inf
    for held_eval, x, after, _score in ranked[:HELD_TOP]:
        value = held_eval + NEXT_DISCOUNT * _best_next_score(
            after, obs.next_type, step=NEXT_CANDIDATE_STEP, pool=pool
        )
        if value > best_score:
            best_score = value
            best_x = x
    return best_x


def _candidates(
    fruits: tuple[Fruit, ...] | list[Fruit],
    drop_type: int,
    held_r: float,
    extra_type: int | None = None,
    *,
    step: float | None = None,
) -> list[float]:
    """Uniform spacing plus spots above / beside same-type and nearby fruits, and ideal_x."""
    sign = _order_sign(fruits)
    lo = held_r
    hi = NORMALIZED_WIDTH - held_r
    grid = CANDIDATE_STEP if step is None else step
    # List every multiple of grid within lo..hi.
    xs = {i * grid for i in range(math.ceil(lo / grid), int(hi / grid) + 1)}
    xs.add(pen.ideal_x(drop_type, sign))
    _add_near_fruit_x(xs, fruits, held_r, lambda t: drop_type <= t <= drop_type + 2)

    if extra_type is not None:
        xs.add(pen.ideal_x(extra_type, sign))
        _add_near_fruit_x(xs, fruits, held_r, lambda t: t == extra_type)

    return [x for x in xs if lo <= x <= hi]


def _add_near_fruit_x(
    xs: set[float],
    fruits: tuple[Fruit, ...] | list[Fruit],
    held_r: float,
    matches,
) -> None:
    """Add the positions above / touching left and right of fruits whose type satisfies matches to xs."""
    for fruit in fruits:
        if not matches(fruit.type):
            continue
        xs.add(fruit.x)
        gap = held_r + fruit.radius
        xs.add(fruit.x - gap)
        xs.add(fruit.x + gap)


def drop_scores(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    *,
    next_type: int | None = None,
) -> tuple[float, float, float, list[Fruit], int]:
    """(score, penalties, eval, after, merges) of one move dropped at column x.

    For sim / training. after and merges are the simulate_drop results as is.
    Board penalties are the difference from before the drop. On the same board it is a constant difference, so choose_x's
    choice does not change, and summing it per move does not grow with the size of the board.
    """
    held_r = fruit_radius(drop_type)
    before = list(fruits)
    after, score, penalties, merges, held_merged = _evaluate_drop(
        before,
        drop_type,
        clamp_drop_x(x, drop_type),
        held_r,
        next_type=next_type,
    )
    # Subtract the pre-drop part on the same basis as the post-drop part. Excluding size_order on the after side
    # while subtracting it included on the before side subtracts a penalty that does not exist, and merge moves
    # become unfairly favored (a measured offset of 0.303).
    penalties -= pen.board_penalties(
        before, sign=_order_sign(before), exempt_size_order=held_merged
    )
    return score, penalties, score - penalties, after, merges


def _held_eval(
    obs: Observation, x: float, held_r: float
) -> tuple[list[Fruit], float, float]:
    """(board, score - penalties, real-game score) after dropping held at x. Does not look at next."""
    assert obs.held_type is not None
    before = list(obs.fruits)
    after, score, penalties, _merges, _held_merged = _evaluate_drop(
        before, obs.held_type, x, held_r, next_type=obs.next_type
    )
    return after, score - penalties, score


def _score(obs: Observation, x: float, held_r: float) -> float:
    """Score the board after dropping held + the hypothetical best move of next."""
    after, value, _score = _held_eval(obs, x, held_r)
    if obs.next_type is not None:
        value += NEXT_DISCOUNT * _best_next_score(after, obs.next_type)
    return value


def _next_eval_job(fruits: list[Fruit], next_type: int, next_r: float, nx: float) -> float:
    """eval for one next candidate. The unit sent to the pool."""
    # The next after that is unknown. Only same-type fruit in a valley counts for the growing exemption.
    _, score, penalties, _merges, _held_merged = _evaluate_drop(
        fruits, next_type, nx, next_r
    )
    return score - penalties


def _best_next_score(
    fruits: list[Fruit],
    next_type: int,
    *,
    step: float | None = None,
    pool: Executor | None = None,
) -> float:
    """eval when next is dropped at its best column."""
    next_r = fruit_radius(next_type)
    xs = [clamp_drop_x(nx, next_type) for nx in _candidates(fruits, next_type, next_r, step=step)]
    if not xs:
        return 0.0
    if pool is None:
        scores = [_next_eval_job(fruits, next_type, next_r, nx) for nx in xs]
    else:
        scores = list(
            pool.map(
                _next_eval_job,
                itertools.repeat(fruits),
                itertools.repeat(next_type),
                itertools.repeat(next_r),
                xs,
            )
        )
    return max(scores)


def _evaluate_drop(
    fruits: list[Fruit] | tuple[Fruit, ...],
    drop_type: int,
    x: float,
    held_r: float,
    *,
    next_type: int | None = None,
) -> tuple[list[Fruit], float, float, int, bool]:
    """Board, real-game score, penalties, merge count and whether held merged after one drop."""
    before = list(fruits)
    sign = _order_sign(before)
    after, merges, merge_types, held_merged, held_fruit = simulate_drop_held(
        before, drop_type, x
    )
    land_x, _land_y = landed_xy(before, after, drop_type, x, held_r, held_merged)

    score = merge_score(merge_types)
    # When held (this move) merged, size-order and burying penalties are not applied to unrelated fruits
    # knocked by its recoil. Looking only at `merges >= 1` would also exempt merges that happened by chance elsewhere
    # on the board unrelated to held,
    # so judge by whether held itself took part in a merge (`held_merged`).
    penalties = pen.board_penalties(after, sign=sign, exempt_size_order=held_merged)
    # FOREIGN_AIM looks at 'is the fruit directly below a different type', not merges.
    # A same type directly below is OK (waiting to merge). Rolling off a different type and merging on the floor is still penalized.
    penalties += pen.foreign_aim_penalty(before, x, drop_type, held_r)
    # A term only for breaking ties. It decides the order when every term above ties.
    penalties += pen.center_tiebreak(x)
    if not held_merged:
        # Valley growing. Among non-merging moves, choose landings in valleys likely to grow.
        # Merging moves get the real-game score, so it is not added to them.
        penalties -= pen.valley_grow_bonus(before, land_x, drop_type, next_type)
    else:
        # Which way the fruit made by the merge went. Merging moves are exempt from size order
        # (exempt_size_order), so no other term looks at which side it was hit from and where the new fruit was thrown.
        # At the same merge score, choose the way of hitting that pushes toward the big side.
        penalties -= pen.merge_big_side_bonus(x, held_fruit, held_r, sign)
    return after, score, penalties, merges, held_merged


def _order_sign(fruits: list[Fruit] | tuple[Fruit, ...]) -> int:
    """The size direction of the board. +1 = big left, small right; -1 = small left, big right."""
    if not fruits:
        return 1
    if len(fruits) == 1:
        fruit = fruits[0]
        if fruit.type >= SPAWN_MAX_TYPE and fruit.x > NORMALIZED_WIDTH * 0.55:
            return -1
        return 1

    votes = 0.0
    for i, a in enumerate(fruits):
        for b in fruits[i + 1 :]:
            if a.type == b.type:
                continue
            if abs(a.x - b.x) < min(a.radius, b.radius) * 0.5:
                continue
            left, right = (a, b) if a.x <= b.x else (b, a)
            weight = float(abs(a.type - b.type)) * (1.0 + 0.15 * max(a.type, b.type))
            if left.type > right.type:
                votes += weight
            else:
                votes -= weight

    if abs(votes) < 1.0:
        biggest = max(fruits, key=lambda f: (f.type, f.radius))
        return -1 if biggest.x > NORMALIZED_WIDTH * 0.5 else 1
    return 1 if votes > 0 else -1
