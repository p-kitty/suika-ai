# Known issues and things to fix later

## Contents

- [Open tasks](#open-tasks)
- [In progress: big draws and ladders after the floor fills](#in-progress-big-draws-and-ladders-after-the-floor-fills)
- [How to measure](#how-to-measure-traps-we-keep-stepping-in) ← read before reporting numbers
- [Investigated: sudden death from scattered low-tier fruits late in the game](#investigated-sudden-death-from-scattered-low-tier-fruits-late-in-the-game)
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
- Friction between fruits seems low: fruits slide in far more than in the real game.

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

## How to measure (traps we keep stepping in)

**Score noise is very large.** This is the biggest wall for improving the policy, and every attempt below
got buried here. Read this section before reporting numbers.

- The per-game standard deviation is ~1000-1200, and even in paired comparisons on the same seed the SD of the difference is 78-496.
  Seeing ±100 points as significant needs **n≈100**. Tens of episodes are not enough.
  it is faster to look for a proxy metric with lower variance than score (moves survived, the number of isolated fruits in specific positions and so on)
- **Discard the numbers if every episode reaches `max_steps`.** Neither headroom nor death rate is measured.
  Natural ends are 300-400 moves, so `--max-steps` around 400. `compare_policy.py` warns
- Do not fix seeds (omitting `--seed` makes them random). Reusing fixed seeds makes a chance collapse
  easy to misread as "reproduced". Compare changes paired on the same seeds
- Do not build automation scripts that run `git stash` / `checkout` against the live working tree.
  It once caused an accident that left uncommitted changes stranded in the stash (recovered with `git fsck --unreachable`).
  To compare another commit with the current working tree, isolate it with `git worktree add`

## Investigated: sudden death from scattered low-tier fruits late in the game

A record of the investigation when trying to strengthen bootstrap toward around 3500 points.

### Diagnosing the cause of death

Measured with eval_policy.py at 24-40 episodes, the mean score is 1900-2150 points
(the "one game of 311 moves, score 3305" at the top of NOTES was one favorable example, not a typical result).
Every episode ends in `dead` and never reaches the `max_steps` cap.

Replaying one episode move by move (seed=20260816), for 10+ moves before death
low-tier fruits such as 5-8 cherries, 3 grapes and 4 dekopons remained scattered unmerged
across the whole board. The final move (an orange) had no safe landing and was forced onto the tower at the upper right,
already at a dangerous height, dying instantly. It was further confirmed that in the position just before, dropping a cherry anywhere from x=0-400
produces no merge at all. So it is not a problem with "that move":
much earlier, low-tier fruits were squeezed from the sides by big fruits of other types and scattered into
physically unmergeable positions, which is the root cause.

### Improvement attempts (none confirmed; code reverted)

| Attempt | Content | Result |
|---|---|---|
| triangular excess-same | `_excess_same_penalty` from linear (1 excess = 20) to triangular (1 excess = 1x, 2 = 3x…) | results split: +1.9% at n=24 / -3.0% at n=40 |
| side isolation penalty | a new `_isolation_penalty` detecting moves that block both sides of a partnerless fruit | -9.2% at n=32 (difference 195 against SE~206) |
| three-ply lookahead | expand the third move only for the top `NEXT_TOP=2` by eval. The type is unknown, so approximated by the mean of one `ideal_x` point over 5 types, `THIRD_PLY_DISCOUNT=0.4`. 313→401ms per move | -0.7% at n=32 |

Symptomatic fixes (weight tuning, new penalties, deeper lookahead) were buried under the noise floor
all three times. On the RL side too, [BC does not reach 60-70% match](#investigated-bc-does-not-reach-60-70-match-2026-08-05)
showed a plateau, and fine-tuning bootstrap or extending shallow lookahead gives no
outlook toward 3500 points. What to try next would need 100+ episodes of re-verification, or
a qualitatively different change such as rebuilding the features and architecture of the learned policy itself.

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
| blocking a waiting merge by burying | `_bury_block_penalty` | when a bigger fruit of another type blocks, directly above or on the shoulder, a fruit waiting for a same-type pair | 14.0 ×type gap (half on a shoulder) |
| small-side escape after the floor fills | `_packed_small_side_penalty` | after the floor packs, when a large draw (orange or bigger) escapes to the small side (fires only when it physically cannot go on the small side) | fixed 8.0 (can be disabled with `SUIKA_PACKED=0`) |

The 2 below apply **only when held itself did not merge** (`held_merged`, not the merge count
`merges`, so that an unrelated merge elsewhere on the board does not grant the exemption).

**Board-wide penalties (`_board_penalties`, on the post-drop board every time)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| dangerous height | inline | when the topmost crown is above danger_y(70.9) | (danger_y − crown) × 0.5 |
| burying | `_bury_penalty` | how much merge-candidate fruits are covered by other types (with sibling 1.0 / without 0.35) | bury_weight 20.0x |
| excess same type | `_excess_same_penalty` | 3 or more of the same type (up to 2 are allowed as waiting to merge) | 20.0 per excess fruit |
| size-order inversion | `_size_order_penalty` | pairs whose size order is inverted left to right (fruits being grown in a valley are exempt). **Exempt on moves where held merged** (so unrelated fruits knocked by merge recoil are not counted as violations) | pair difference×1.5 + ideal_x deviation×0.004 |
| big-fruit layout | `_big_layout_penalty` | (1) the biggest fruit is on the big-side wall yet a small fruit is outside and below it (corner pocket filled) (2) big fruits not close enough | (1) 50.0×(1+0.05×type gap)+depth×0.15  (2) gap×0.025×size factor |
| bumpiness (height variance) | `_height_variance` | spread of crown heights per column bin (scaled by 0.15 at dangerous height) | variance×0.08 |

Notes:
- `PACKED_RULE_ENABLED` (environment variable `SUIKA_PACKED`) toggles only the small-side escape penalty after the floor fills ON/OFF (for A/B)
- Ladder detection (`_ladder_*`) is currently unused by penalties (detection only)

### Exempting held merges (`held_merged`) — effect unconfirmed (2026-08-06)

`_size_order_penalty` counted layouts knocked by merge recoil as violations, and there were positions where **merging moves
lost to non-merging moves** (a grape between two pears with a gap of 40: merge score 6.0 with
a penalty of 4.94 on top, reversing it). `simulate_drop_held` tracks whether held's lineage took part in a merge, and
moves where it did are now exempt. See the code comments for the mechanism.

**There is no score support. Whether to keep or remove it is undecided.**

- n=20 paired comparison (same seeds, `max_steps=400`): 2046.9 → 1942.9
  (**-104.0**, t=-0.94, 95%CI -326 to +118, 7 wins 13 losses). Not significant, but the point estimate is negative.
  A truncated run on other seeds also gave -40.4 / 7 wins 13 losses, negative both times. **No evidence it got stronger**
- The firing volume is small too. Over 1001 measured candidates, `merges>0 and held_merged=False` is **0**,
  and the exempted `_size_order_penalty` is median 0.19 / max 1.77 (merge scores are 1-65)

Two existing inconsistencies fixed at the same time (these are correct fixes regardless of score):

- `landed_xy` cut on `merges`, so when merely an unrelated merge happened it returned a geometric estimate instead of
  the actual resting position, and `_bury_block_penalty` acted on false coordinates (a measured offset of 149.6)
- `drop_scores` excluded `_size_order_penalty` on the after side while subtracting it included on the before side,
  subtracting a penalty that did not exist (measured 0.303)

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

### Investigated: BC does not reach 60-70% match (2026-08-05)

When trying to move on to RL aiming for 3500 points, the existing checkpoints
(`artifacts/policy_sim.npz` / `policy_sim_rl.npz`) were measured first:
score was only about half of bootstrap (~2000-2150) (~1040-1060),
far from the condition for starting RL (match 60–70%+).

**Bug found (fixed)**: `MAX_FRUITS` in `src/encode.py` was 16,
packing the board's fruits starting from the biggest types and ruthlessly cutting off the rest.
Late in the game boards with over 20 fruits are common (23 measured), and the first to be cut are
scattered low-tier fruits such as cherry/strawberry —
exactly the culprits of the accident identified in [sudden death from scattered low tiers late](#investigated-sudden-death-from-scattered-low-tier-fruits-late-in-the-game)
were invisible to the student. Raised to 32 so effectively
nothing is cut.

**What it still did not solve**: retraining BC after the fix with default settings (100 collection ep, 80 epochs)
gave match 12.9% and score only 1242 (loss barely moved across epochs,
clear under-fitting). The teacher data (150 ep, n=32562) was cached and
reused to compare the following on the same data; none reached match 30%, and score
plateaued at 900-1150:

- an exhaustive sweep of lr 0.05-1.0 × epoch 80-300 × hidden 128/256 (best: hard label,
  lr=0.5, hidden=256, epoch=300 with match=29.1%)
- comparing a hard label approach that one-hots the teacher's continuous x as is vs a soft label approach that also
  spreads weight to nearby bins (`teacher_action_target` / `bc_update_dist`,
  which existed in the code but were unused by the training pipeline). Even soft stopped
  below the best hard (15-21%)

**What we learned and hypotheses**: bootstrap is a policy that actually runs `simulate_drop` per candidate
and compares the results, and imitating from only the static features of `src/encode.py` (a list of fruit type/x/y/r
a 1-hidden-layer MLP learning that "try candidates with physics, then choose" judgment
plateaued whichever of learning rate, epochs and soft/hard labels was tuned.
Unless the capacity (hidden width, layer count) or the feature design (such as including per-candidate landing results
in the features) itself changes, BC in this configuration is likely not to reach
the condition for starting RL. RL remains unstarted.
