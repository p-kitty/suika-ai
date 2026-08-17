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

**Physics simulation**
- Friction between fruits seems low: fruits slide in far more than in the real game.

**Training pipeline**
- Training episode length: raise `max_steps` and lower `episodes` (fewer, longer games).
  The default in `train_sim.py` is 300 (measured natural ends are median 210 moves and max 311, so
  about 2% are truncated. 320 for 0%)

## In progress: big draws and ladders after the floor fills

After the floor fills, placing big draws such as orange / dekopon on the small side crushes the fruits below and the board collapses.
Handled by `_packed_small_side_penalty` (`src/policy.py`). **Always on** (the toggle is removed).

### Result at n=100 (2026-08-06): no significant difference

`--episodes 100 --max-steps 400` (0 truncated, 2.6 hours):

- score 2047.0 → 2093.9 (**+46.9**, t=0.85, 95% CI **[-62.3, +156.1]**). **Not significant**
- seed head-to-head win 52 / loss 38 / tie 10. With 10 ties, there are 90 firing opportunities
- **By quantile only the bottom rose** (min 1045→1315, bottom-10 mean 1318→1450).
  The top does not move (top-10 mean 2914→2913, type10 reached 14→17 runs).
  It is a rule against collapse after the floor fills, so this matches the intent, but **choosing the bottom after the fact
  and testing it is post-hoc selection**. Next time, fix a threshold such as "number of runs with score<1500" before measuring
- Making this +46.9 significant needs **n=529 (about 14 hours)**. Not worth it, so without remeasuring it was
  **made permanent as ON** (positive point estimate, shape matches the intent, fires 90/100). The A/B toggle
  (`SUIKA_PACKED` / `set_packed_rule_enabled`) is removed

The "ladder" that fires a corner big fruit in steps (a pear next to the inside of a corner peach, an apple and an orange on the **shoulders** of those two,
firing with the final orange and cascading 4→5→6→7) is a shape that arises naturally as a result of this placement rule.
Only detection is written in `find_anchor` / `rungs` of `src/ladder.py`, and **it has never been called from the production path**
(only `tests/test_policy.py` calls it). It is kept as groundwork for using it in move selection.

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
- **Do not judge by a rise or fall in the mean alone.** `compare_policy.py` prints, per metric, the paired t value and
  95% CI (`src/stats.py`). A row whose CI crosses 0 says nothing at that n.
  When not significant it also shows "the n needed to speak to ±100 points"
- Add `--out artifacts/xxx.json` to long runs to keep per-seed raw data.
  It is written before aggregation, so a bug on the aggregation side does not lose hours of work
- Proxy metrics are chosen with `scripts/analyze_ab.py <dump>`. It ranks metrics by n_detect (the number of episodes
  needed to move that difference away from 0). It is unit-independent, so score and moves survived can be
  compared directly. But **picking the metric that looked best in the same dump is selection bias**.
  Adopt it only after confirming it also ranks high on a second dump taken with a different change and different seeds
- **Discount the numbers when truncation happens.** Truncated games are the ones that went long, so
  the better the change the more it is underestimated. Natural ends were **measured over 200 runs: mean 213 / median 210 / max 311 moves**
  (the old "300-400 moves" came from one favorable game and was an overestimate).
  `--max-steps` **truncates 0% at 320 or more**; the default is 400. At 200 it is 64%,
  **at 100, 100% are truncated** (the 08-03 accident was this). `compare_policy.py`
  warns if even one is truncated
- **Do not stratify by the outcome and compare the same outcome (regression to the mean).** Splitting into top/bottom by A's score
  and comparing A with B always shows "the top got worse, the bottom improved". Splitting by B's score
  gives the mirror image. This was nearly stepped on with the n=100 of `SUIKA_PACKED`.
  To compare distributions, compare quantiles directly
- Do not fix seeds (omitting `--seed` makes them random). Reusing fixed seeds makes a chance collapse
  easy to misread as "reproduced". Compare changes paired on the same seeds

The measuring procedures themselves (how to plug in an A/B is in [AGENTS.md](AGENTS.md#when-touching-the-policy-or-training),
comparisons that do not dirty the working tree are in [git in AGENTS.md](AGENTS.md#git)) live there.
Only what can be trusted is written here.

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
- Valley growing (`_valley_grow_ok`): the valley fruit is the same type as held, or the valley fruit is one above held and
  held and next are the same type: a bonus of `VALLEY_GROW_BONUS` for landing in that valley. The reference is the valley fruit;
  the wall types are not looked at. Other gap filling gets the usual penalties (`GAP_JUNK` stays retired)
- Layout: big fruits stay close together. On the big side (`sign`), the corner pocket outside an edge-anchored L and below L's center
  is heavily penalized (`_big_layout_penalty`)
- But if a type is missing between two neighbors, that much gap is not closed. Closing it leaves
  no place for the missing type when it is drawn, and the only option is to send it outside and break the order
  (pulling a grape right beside an opening orange made the next dekopon fall outside, giving 4-2-3)
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
| small-side escape after the floor fills | `_packed_small_side_penalty` | after the floor packs, when a large draw (orange or bigger) escapes to the small side (fires only when it physically cannot go on the small side) | fixed 8.0 |
| valley-growing bonus | `_valley_grow_ok` | landing in a valley whose fruit is the same type as held / whose fruit is one above held with held and next the same type | **−3.0** (`VALLEY_GROW_BONUS`. The only bonus in this table) |

The 3 below apply **only when held itself did not merge** (`held_merged`, not the merge count
`merges`, so that an unrelated merge elsewhere on the board does not grant the exemption).

**Board-wide penalties (`_board_penalties`, on the post-drop board every time)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| dangerous height | inline | when the topmost crown is above danger_y(70.9) | (danger_y − crown) × 0.5 |
| burying | `_bury_penalty` | how much merge-candidate fruits are covered by other types (with sibling 1.0 / without 0.35) | bury_weight 20.0x |
| excess same type | `_excess_same_penalty` | 3 or more of the same type (up to 2 are allowed as waiting to merge) | 20.0 per excess fruit |
| size-order inversion | `_size_order_penalty` | pairs whose size order is inverted left to right (fruits stuck in a valley of bigger fruits are exempt = `_is_nestled`. Whether growing succeeds is not checked). **Exempt on moves where held merged** (so unrelated fruits knocked by merge recoil are not counted as violations) | pair difference×1.5 + ideal_x deviation×0.004 |
| big-fruit layout | `_big_layout_penalty` | (1) the biggest fruit is on the big-side wall yet a small fruit is outside and below it (corner pocket filled) (2) big fruits not close enough (exempt for the diameter of the missing type in between) | (1) 50.0×(1+0.05×type gap)+depth×0.15  (2) (gap−diameter of the missing type)×0.025×size factor |
| bumpiness (height variance) | `_height_variance` | spread of crown heights per column bin (scaled by 0.15 at dangerous height) | variance×0.08 |

Notes:
- The rules above have no ON/OFF toggles (the policy is not to keep toggles for permanent rules.
  The A/B procedure is in [AGENTS.md](AGENTS.md#when-touching-the-policy-or-training))
- Ladder detection (`src/ladder.py`) is currently unused by penalties (detection only)

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
  per-seed wins and losses, per-phase metrics (`early_score` / `early_crown` / `dead_early` / `cascades`), and
  per-metric paired t values and 95% CIs. **Plug the change you want to compare into `_apply_variant`**
  (empty makes A and B identical, and a warning that every seed tied appears). A warning appears if even one `max_steps`
  A warning appears on truncation. Omitting `--seed` makes it random (reusing fixed seeds invites misreading).
  `--out` saves raw data as JSON
- Searching for proxy metrics: `python scripts/analyze_ab.py <dump.json>` (→[How to measure](#how-to-measure-traps-we-keep-stepping-in))
- Statistics are in `src/stats.py` (paired t / 95% CI / required n / correlation). scipy is not installed
- Training: `python scripts/train_sim.py` (collect → offline BC. The default max-steps=300 is a cap,
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
