# Known issues and things to fix later

## Contents

- [Open tasks](#open-tasks)
- [In progress: big draws and ladders after the floor fills](#in-progress-big-draws-and-ladders-after-the-floor-fills)
- [When to move](#when-to-move)
- [Policy (bootstrap) design](#policy-bootstrap-design)
- [Training](#training)
- [Planned: RL (REINFORCE)](#planned-rl-reinforce)

## Open tasks

**Vision**
- `10.png`: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than policy / held

**Policy behavior**
- Packing too tight early: neighbors by size order and proximity (e.g. apple and dekopon) have too little gap between them. When stacking an orange only order-breaking moves remain. Some space is wanted

**Physics simulation**
- Wall friction may be too low: at the walls in `sim_physics.py`, fruits slide in far more than in the real game
- Friction between fruits also seems low: same as above. Fruit-to-fruit contact feels slipperier than the real game too. Revisit both wall and fruit friction coefficients together

**Training pipeline**
- Training episode length: raise `max_steps` and lower `episodes` (fewer, longer games). Guide: natural ends around 300-400 moves (measured one game at 311 moves, score 3305, type 10 reached). Truncating at 100-250 moves cannot measure headroom or survival time

## In progress: big draws and ladders after the floor fills

After the floor fills, placing big draws such as orange / dekopon on the small side crushes the fruits below and the board collapses.
Being handled by `_packed_small_side_penalty` (`src/policy.py`). Can be disabled with `SUIKA_PACKED=0`
(`set_packed_rule_enabled()`). **Its effectiveness is not yet established** (below).

The "ladder" that fires a corner big fruit in steps (a pear next to the inside of a corner peach, an apple and an orange on the **shoulders** of those two,
firing with the final orange and cascading 4→5→6→7) is a shape that arises naturally as a result of this placement rule.
Only detection runs in `_ladder_anchor` / `_ladder_rungs`, not used for move selection (no flag, always computed).

### What we know

- **Firing needs no guidance**. Once a ladder is built, `choose_x` ties with the best of an exhaustive sweep over x.
  Adding candidate x for firing was a no-op (removed)
- **`FOREIGN_AIM` is unrelated to the ladder**. The rungs sit on shoulders, so this penalty never applies in the first place.
  It is physically never stable directly on top
- **`_size_order_penalty` is not in the way either**. Measured on ladder boards it is only 0.14-0.49
- **Without a filled floor the shape does not hold**. The pear is pushed out like a wedge and self-destructs, and wherever you drop
  you get only one rung (15 points). Filling the floor to the right edge gives 100 points. A filled floor is a gate condition
- **The bottleneck is building it**. A board with all 4 rungs appears only 12 times in 720. Writing a rung as a condition of "a move
  that drops and places it" is a poor approach (draws go up to orange; pear and apple can only be made by merging)
- The small-side room check confirms with an actual `simulate_drop`, not just the geometric gap width
  (`_small_side_room_ok`). Fixed a bug that judged a gap blocked by a roof as having room

### Measurement caveats (today's lesson)

- `compare_policy.py` detects truncation (every episode reaching `max_steps`) and warns.
  Do not ignore it and draw conclusions
- Do not fix seeds (omitting `--seed` makes them random). Reusing fixed seeds makes a chance collapse
  easy to misread as "reproduced"
- About 4 episodes cannot decide anything. Measured (4 runs on the same seeds, comparing the 3 states bf8ea23 / 5be107f / working tree
  isolated with `git worktree`): the score difference is buried in per-game variation (SD ~1168),
  and a significant conclusion needs tens of runs
- Do not build automation scripts that run `git stash` / `checkout` against the live working tree.
  Once, an accident left uncommitted changes stranded in the stash (`git fsck --unreachable` could
  recover it, but when comparing historical code of another commit with the current working tree,
  isolate it with `git worktree add`)

## When to move

- Do not decide x on a moving board. Waiting for it to settle takes priority over lookahead (`src/settle.py`)
- Wait for creep not only on instantaneous velocity but also on sideways drift while quiet

## Policy (bootstrap) design

- `src/policy.py` is a thin policy before RL. Only merging, dangerous height, burying, light size order and accident prevention for rolling / knock-aways
- The physics of falling, collision and merging is pymunk (`src/sim_physics.py`; UT in `tests/test_sim_physics.py`).
  `choose_x` scores with the same `simulate_drop`
- Moves are scored as `eval = score - penalties`. The only bonus is the real game's score; dangerous height, accidents and burying are penalties
- next lookahead: only the top `HELD_TOP` by held eval are re-evaluated with candidates at spacing `NEXT_CANDIDATE_STEP`
  (multiplied by `NEXT_DISCOUNT`). The physics is heavy, so it is coarser than held
- Burying is the main penalty. Moves that block a same-type pair waiting to merge with a bigger fruit of another type, directly above or on the shoulder, are heavily penalized
- Aiming at the center of a different type (`FOREIGN_AIM`): if the fruit directly below is a different type and in its center band (`FOREIGN_AIM_CENTER_FRAC`),
  `FOREIGN_AIM_PENALTY`. OK if it is the same type. Not cut by `merges` (closes the loophole of rolling off a different type
  and merging). Stacking a different type in valleys or on shoulders is not itself forbidden (this penalty is only the center band)
- Excess same type (`EXCESS_SAME`): once 3 or more of a type accumulate, a penalty of 20 per excess fruit
- Valley growing for big fruits is limited to when the valley has a same type, or held/next are both one smaller than the walls.
  Other gap filling gets the usual penalties (`GAP_JUNK` stays retired)
- Layout: big fruits stay close together. On the big side (`sign`), the corner pocket outside an edge-anchored L and below L's center
  is heavily penalized (`_big_layout_penalty`)
- Rolling accidents onto the big-side floor (`_wrong_side_roll_penalty`): penalized when rolling into the big-side floor of a big fruit.
  A separate function from the layout penalty above, but with the same aim of preventing big-side accidents
- Not included: push-in merges, restoring pushes, cascade gap opening, forced moves one tier up, hard-coded ladder firing
- Do not add UTs for concrete procedures. When something breaks, look at accident prevention or the observation side
- The search cost is essentially the number of `simulate_drop` calls. `HELD_TOP` / `NEXT_CANDIDATE_STEP` decide the run time
  decide the run time (the old 8/16 took 3.8 seconds per move and collection could not keep up. 2/32 gave 1.2 seconds and score -3.4%)
- Do not make `CANDIDATE_STEP` coarser. At 20 the spot directly above a dangerous pile lands on the grid and
  `test_avoids_dangerous_tall_stack` fails. Speed is earned on the lookahead side
- Cutting `SLEEP_FRAMES` does not work. A single `choose_x` gets faster, but the board settles differently and
  later moves get heavier, so the whole episode is actually slower (measured at 25). The physics fidelity
  (shared with `SimEnv`) also drops

### Current penalty rules

`eval = score (merge points) - penalties (penalties for accidents and bad moves)`.
- per-move penalties in `_evaluate_drop`,
- board-wide penalties in `_board_penalties` (`src/policy.py`) are each added.

**Per-move penalties (`_evaluate_drop`)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| directly above a different type | `_foreign_aim_penalty` | when the fruit directly below the drop column (center offset within ±20%) is a different type | fixed 100.0 |
| wrong-side roll | `_wrong_side_roll_penalty` | when it rolls and lands on the floor on the "big side" of a bigger fruit of another type (only moves with 0 merges and outside the growing exemption) | base 8.0 + difference×2.0 |
| blocking a waiting merge by burying | `_bury_block_penalty` | when a bigger fruit of another type blocks, directly above or on the shoulder, a fruit waiting for a same-type pair | 14.0 ×type gap (half on a shoulder) |
| small-side escape after the floor fills | `_packed_small_side_penalty` | after the floor packs, when a large draw (orange or bigger) escapes to the small side (fires only when it physically cannot go on the small side) | fixed 8.0 (can be disabled with `SUIKA_PACKED=0`) |

**Board-wide penalties (`_board_penalties`, on the post-drop board every time)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| dangerous height | inline | when the topmost crown is above danger_y(70.9) | (danger_y − crown) × 0.5 |
| burying | `_bury_penalty` | how much merge-candidate fruits are covered by other types (with sibling 1.0 / without 0.35) | bury_weight 20.0x |
| excess same type | `_excess_same_penalty` | 3 or more of the same type (up to 2 are allowed as waiting to merge) | 20.0 per excess fruit |
| size-order inversion | `_size_order_penalty` | pairs whose size order is inverted left to right (fruits being grown in a valley are exempt) | pair difference×1.5 + ideal_x deviation×0.004 |
| big-fruit layout | `_big_layout_penalty` | (1) the biggest fruit is on the big-side wall yet a small fruit is outside and below it (corner pocket filled) (2) big fruits not close enough | (1) 50.0×(1+0.05×type gap)+depth×0.15  (2) gap×0.025×size factor |
| bumpiness (height variance) | `_height_variance` | spread of crown heights per column bin (scaled by 0.15 at dangerous height) | variance×0.08 |

Notes:
- `PACKED_RULE_ENABLED` (environment variable `SUIKA_PACKED`) toggles only the small-side escape penalty after the floor fills ON/OFF (for A/B)
- Ladder detection (`_ladder_*`) is currently unused by penalties (detection only)

## Training

- **score / penalties**: `score` is the real game's merge score (1-65, no penalties), `penalties` are the penalties for accidents and bad moves.
  `eval = score - penalties` is **only for bootstrap move selection** (`choose_x`). The student's quality, saving and
  logs use the real game's `score` (moves on ties). **The RL reward is score too** (dense penalties are not rewards)
- `src/reward.py`: `merge_score(merge_types)` gives only merge points identical to the real game (cherry→0 …
  watermelon 55, double clear 65). No survival bonus or death penalty. Episodes end as before
  (losing line / double clear)
- `src/encode.py`: fixed-length observation vector
- `src/sim_env.py`: headless drop sim (`sim_physics.simulate_drop`). `SimStep` is the real game's `score`
  only (no cumulative eval)
- Evaluation: `python scripts/eval_policy.py` (`--policy bootstrap|learned`. `--workers` default = logical cores/2)
- A/B: `python scripts/compare_policy.py`. Pits two bootstrap variants against each other, reporting not just means but
  per-seed wins and losses and per-phase metrics (`early_score` / `early_crown` / `dead_early` / `cascades`)
  are reported. If every seed ties it warns "the change is not firing", and if every episode is truncated at `max_steps`
  it warns about that too. Omitting `--seed` makes it random (reusing fixed seeds invites misreading)
- Training: `python scripts/train_sim.py` (collect → offline BC. The default max-steps=100 is a cap
  not the losing line). best is score → moves → match
- Teacher collection runs in parallel with `ProcessPool` (default workers=logical cores/2; 8 on a 9700X; `--workers 1` for serial)
- `src/agent.py`: MLP with 32 discrete column bins / hidden 128 (old 20/64 npz files need retraining)
- Live play: `python main.py` (defaults to learned if an npz exists. `L` toggles bootstrap, `--policy bootstrap`)

## Planned: RL (REINFORCE)

- Still too early. Plain REINFORCE easily breaks things when BC is shallow (confirmed in the past)
- Condition for adding it: `match` fairly high (roughly 60–70%+) and the student's `score` close to bootstrap
- How: only a short fine-tune after BC finishes (e.g. `--episodes 50 --lr 0.002`). Off by default
- Until then, thickening BC (collection size, epochs) comes first
