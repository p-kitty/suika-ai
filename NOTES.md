# Known issues and things to fix later

## Contents

- [Open tasks](#open-tasks)
- [In progress: big draws and ladders after the floor fills](#in-progress-big-draws-and-ladders-after-the-floor-fills)
- [How to measure](#how-to-measure-traps-we-keep-stepping-in) ← read before reporting numbers
- [Investigated: sudden death from scattered low-tier fruits late in the game](#investigated-sudden-death-from-scattered-low-tier-fruits-late-in-the-game)
- [Investigated: how the board collapses and isolating the stage](#investigated-how-the-board-collapses-and-isolating-the-stage-2026-08-18)
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
  95% CI (`src/util/stats.py`). A row whose CI crosses 0 says nothing at that n.
  When not significant it also shows "the n needed to speak to ±100 points"
- Add `--out artifacts/xxx.json` to long runs to keep per-seed raw data.
  It is written before aggregation, so a bug on the aggregation side does not lose hours of work
- Proxy metrics are chosen with `scripts/analyze_ab.py <dump>`. It ranks metrics by n_detect (the number of episodes
  needed to move that difference away from 0). It is unit-independent, so score and moves survived can be
  compared directly. But **picking the metric that looked best in the same dump is selection bias**.
  Adopt it only after confirming it also ranks high on a second dump taken with a different change and different seeds
- **The proxy metric was settled as `cascades` (number of moves with 3+ merges) (2026-08-17).**
  It ranked near the top in all three independent dumps (packed / valley_grow / wider_lookahead),
  stable with sensitivity ratios 1.34 / 1.28 / 1.40 and r(score) 0.83. In wider_lookahead
  **only cascades detected the difference that score could not make significant** (t=2.36).
  Other candidates drop out: `early_crown` is 1st in packed but r≒-0.05, unrelated to score, and
  `steps` / `merges` have a high r=0.97 but about the same sensitivity as score, so no gain.
  It reduces n only by the sensitivity ratio (at 1.4x the required n is about halved), so it is no silver bullet
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

### Investigated: the bonus side of eval is not the bottleneck (2026-08-17)

**Hypothesis** (wrong): the bonus in `eval = score - penalties` is the real game's score itself, so
a cherry merge is only 1 point and even a grape merge only 3. Meanwhile penalties are on the order of `EXCESS_SAME`
20 and `FOREIGN_AIM` 100. So perhaps the motive to clean up small fruits is structurally
buried, leading to the "sudden death from scattered low tiers" above.

**What was tried**: `FRAGMENT_WEIGHT`, penalizing the number of fruits on the board itself
(`w * len(fruits)` in `board_penalties`). One merge lowers the count by 1, so
this is mathematically the same shape as "a merge bonus independent of type". The one fruit always added by the drop
is common to all candidates, so it does not affect move choice, and it applies in the same form to the next move of the lookahead.
It was tried as a change to the shape of the bonus side rather than adding a separate penalty rule.

**It failed before reaching an A/B.** In agreement screening (3 seeds × 60 moves = 180 positions), the fraction choosing
the same x as current was **98.9-98.3% at w=3-15, and still 94.4% at w=25**.
It is nearly a no-op, so score was not measured (judged not worth betting hours).

**Why it does not work (recounted without thinning candidates, 3 seeds × 90 moves = 270 moves)**:

| | |
|---|---|
| moves that merged | 138 (51%) |
| moves that passed up an available merging x | **6 (2%)** |
| moves with no merge available anywhere | 126 (47%) |

**In 96% of positions where a merge is possible, the current policy already takes it.** Even with a 1-point bonus,
merging removes a fruit and lifts the bury, excess-same and height penalties wholesale, so
**lifting penalties was standing in for the bonus**. Thickening the bonus side has nowhere to add to.

**The view of the cause of death is updated too.** Measured by game progress (3 games, 575 moves), the fraction of "moves that can merge" is
47% early → 57% late, and **does not fall late** (the actual merge rate follows it).
Meanwhile the fruit count keeps rising, 4.3 → 17.0. So the late collapse is
not "**it can no longer merge**" but "**merging cannot keep up with supply**".
Each move always adds one fruit, while a single merge removes only one; only cascades
remove several. That `cascades` is consistently the sharpest in [proxy metrics](#how-to-measure-traps-we-keep-stepping-in)
is consistent with this too. If intervening, aiming at **how easily cascades happen**
looks like the better approach.

## Investigated: how the board collapses and isolating the stage (2026-08-18)

6 games were traced and counted per move (deterministic per-position quantities, so not
subject to score noise). The policy of "not breaking the board matters more than score"
was reversed in the weighting of the penalties.

### Symptom

- **The inversion rate of horizontal size order goes from 10% early to 40-50% late**. 50% is complete disorder
- **Vertically it was lawless from the start**. Of 23079 vertically stacked pairs, 47% have "the upper one bigger".
  `_size_order_penalty` only looked at `a.x <= b.x`, and there was not a single vertical rule
- cherry / strawberry are the main culprits (+1.25 / +1.57 pairs per move). Only orange recovers, at −0.93
- **Clean moves are among the candidates. The evaluation rejects them**: for cherry, a non-dirtying move is
  a candidate in 97% of positions, yet it is actually chosen in 64%. 2.85 pairs missed per move

### What was beating size order

Taking (chosen move − clean move) per term in positions where a clean move was rejected:

| Term | difference | rate of being the deciding factor |
|---|---|---|
| `bury_block` | **−7.43** | 35% |
| `FOREIGN_AIM` | −6.12 | 6% |
| valley growing | −1.04 | 35% |
| `sizeord` | −0.03 | 16% |

**`sizeord` effectively does not distinguish dirty moves from clean ones.** Converted, one inversion pair
≈ eval 4.91, whereas `bury_block` is 14.0 at type gap 1 and `FOREIGN_AIM` is 100.0.

- `bury_block` fires on 51-72% of small-side candidates (straw/grape/orange), and
  **66% of the pairs it protects already have a bigger fruit wedged between them and cannot merge**.
  83% are more than 3x the contact distance apart. It was paying a median of 28.0 / max 133.0
  for pairs already dead
- Valley growing: **100% of its 1642 firings land on the big side**. 97% are "a same-type fruit is in the valley,
  but this move does not merge" = moves that stack next to that fruit
- The `_size_order_exempt` exemption is an accomplice. On boards with 16+ fruits, 45-57% of small fruits
  are exempt, but removing every exemption only takes it from 4.91 → 6.20 per pair

### Ideas that did not work (dropped at screening)

- **Raising `size_order_pair_weight` from 1.5 → 9.0**: agreement 88.6%, moves barely change,
  and the inversion increase of changed moves is **+0.11** (no improvement). It is a global statistic, the pair count of the whole board,
  so dropping one fruit is buried in the baseline and does not move. **This line is dead**
- In contrast, a local term counting per move "the inversions the dropped fruit itself creates" has
  agreement 80.7% at w=1.0, and changed moves average **−4.00 pairs**. The difference was local versus global

### Material arithmetic (distance to a double watermelon)

Spawns are uniform over type0-4, so 6.2 cherry units per move. One watermelon = 1024 units,
a double watermelon = 2048 units. Measured (average of 6 games): 1217 units on the board at 192 moves, 59% of what is needed.

But **by area there is enough**. The area needed to hold the same 1024 units is
39k for one watermelon, 57k for two melons, 299k for 64 oranges, 651k for 1024 cherries
(**17x less when consolidated**). The board holds 135k measured at death, so
holding 2048 units as two watermelons (79k) physically fits.
**What is missing is neither material nor moves, only consolidation.**

### Metric: inversion rate is unusable for A/B

Using "how clean the board is" as an A/B metric was measured and refuted (n=24).

- The correlation between the all-move mean inversion rate and score is **+0.36** (the reverse sign, dirtier means higher score).
  The longer it lives the more fruits and inversions there are, so it is confounded with game length
- Removing the confound at a fixed move (move 40/60/80/100) gives r = 0.00-0.12, uncorrelated
- Compare: `steps` r=0.95, `cascades` r=0.76

**Use the inversion rate only as a per-move difference, "how much did one move dirty the board".**
It compares candidates on the same position, a deterministic quantity with no noise, and screens in minutes.
Do not put it into an A/B as a per-episode aggregate. The metric stays `cascades`.

### Running BC on only the top teacher data

**Won't do.** Both have measured reasons.

1. The student does not reach the teacher ([BC does not reach 60-70% match](#investigated-bc-does-not-reach-60-70-match-2026-08-05)).
   It cannot even fit the unfiltered teacher, so narrowing to the top only sharpens the target
2. **`choose_x` is deterministic, so the spread of score between seeds is 100% draw-order luck**
   (n=24, mean 2012, SD 332). Picking the top to imitate means learning "how it played
   when lucky"

If done at all the order is reversed: first make `src/training/encode.py` candidate-conditioned
(feed the `simulate_drop` result of each candidate x as input, and the student only reorders).
If that works, making the teacher side a wide 8/16 search is also a good idea
(→[Re-measuring search width 8/16](#re-measuring-search-width-816-2026-08-17). The 3.68x collection cost is one-off for the teacher).

## When to move

- Do not decide x on a moving board. Waiting for it to settle takes priority over lookahead (`src/game/settle.py`)
- Wait for creep not only on instantaneous velocity but also on sideways drift while quiet

## Policy (bootstrap) design

- `src/policy.py` is a thin policy before RL. Only merging, dangerous height, burying, light size order and accident prevention for rolling / knock-aways
- The physics of falling, collision and merging is pymunk (`src/sim/sim_physics.py`; UT in `tests/sim/test_sim_physics.py`).
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
  (the old 8/16 took 3.8 seconds per move and collection could not keep up. 2/32 gave 1.2 seconds and score -3.4%).
  But the unit cost per call became 2.44x faster on 2026-08-17
  → [Faster physics](#faster-physics-2026-08-17). 8/16 was remeasured after speeding up, but
  **not adopted** → [Re-measuring search width 8/16](#re-measuring-search-width-816-2026-08-17)
- Do not make `CANDIDATE_STEP` coarser. At 20 the spot directly above a dangerous pile lands on the grid and
  `test_avoids_dangerous_tall_stack` fails. Speed is earned on the lookahead side
- Cutting `SLEEP_FRAMES` does not work. A single `choose_x` gets faster, but the board settles differently and
  later moves get heavier, so the whole episode is actually slower (measured at 25). The physics fidelity
  (shared with `SimEnv`) also drops

### Faster physics (2026-08-17)

The unit cost of `simulate_drop` went from **9.42ms → 3.86ms (2.44x)**. Play is completely unchanged
(see the verification below), so score was not measured and need not be.

**Where the time went.** 99.2% of `choose_x` is `simulate_drop` (72 calls per move).
Measuring its contents in wall time, the C physics (`cpSpaceStep`) is only 13%, and
**57.9% was `_find_merge_pair`**. A read-only scan just looking for same-type pairs,
reading the position / velocity of every fruit through pymunk properties every substep.
On a 10-fruit board it was called 55,104 times per drop and **returned None 100% of the time**.

**What was done.**

| Change | effect |
|---|---|
| `_MergeScan`: from the gap of the nearest same-type pair and the max speed, compute "how many more substeps contact is impossible" and skip that many scans | scans 55,104 → 3,717 (1/14.8). 9.42 → 4.43ms |
| read `_all_quiet` from the back (the falling fruit is last, so it stops at the first), flatten `_QuietGate` snapshots, cache the moment of inertia per type | 4.43 → 3.86ms. Only 3% on sparse early boards |

The lower bound on the time for a gap to close, since speed only increases through gravity (`elasticity=0`),
comes from the positive root of `g*T² + 2vT = gap`. Push-out corrections (`space.collision_bias`)
do not show up in `body.velocity`, so a 60px/s margin (`SCAN_SPEED_MARGIN`) on the speed and
a cap of 16 substeps (`MAX_SCAN_SKIP`) are applied.

**How it was verified (for this kind of change, score is not measured).** The x / y of every fruit after each move are recorded
with `repr()` (raw float), together with the x chosen by `choose_x`, score and merge count, and
**compared byte for byte** with the output of a master worktree. 3 seeds × 70 moves and 2 seeds × 170 moves
(15 fruits, score 1633/1669) all matched. If the skip is too long by even one substep,
a merge shifts and every later trajectory changes, so it is a sensitive check, not a loose one.

**Little headroom remains.** The breakdown is physics ~62% / quiet gate 18.5% / scan 10.8% /
board setup 5.9%. Cutting physics means fewer substeps or frames,
which changes play (see the `SLEEP_FRAMES` item too).

### Re-measuring search width 8/16 (2026-08-17)

With the physics 2.44x faster, the settings given up for cost, `HELD_TOP=8` /
`NEXT_CANDIDATE_STEP=16`, were remeasured. **The effect is positive but not significant, and it was judged not worth
the cost, so it was not adopted** (the variant in `_apply_variant` was reverted).

**A cheap screening came first.** If widening does not change moves there can be no score difference,
so the agreement of the x chosen by A and B on the same positions was checked first (deterministic,
no noise, minutes). On 210 positions, **75.2% agreement** (the x difference when they disagree has a median of 24px,
max 248px). One move in four differs, so it was judged worth measuring and went to the A/B.
**Follow this order from now on too.** Minutes of screening before betting hours.

**Result at n=100** (`--max-steps 400`, 0 truncated, 2.4 hours):

- score 2105.7 → 2185.3 (**+79.6 / +3.8%**, t=1.68, 95% CI **[-14.3, +173.5]**). **Not significant**
- But **all 9 metrics leaned toward B**. Only `cascades` +6.6% is significant
  (t=2.36, CI [+0.2, +2.2]). Seed head-to-head win 59 / loss 41 / tie 0 (sign test p≒0.07)
- The point estimate of +3.8% **nearly matches, in an independent measurement,** the old record of "-3.4% for 2/32".
  Two measurements not significant alone point to the same size
- Significance needs n≈136. It was within reach, 36 more runs (about 50 minutes)
- **A direct quantile comparison shows it lifts the bottom and trims the top**:
  min 1148→1499, 10% 1531→1816, median 2076→2179, 90% 2722→2620, max 3489→3215.
  Widening the lookahead reduces accidents but also makes big runs less likely. But **type10 reached rose from 12→20 runs**,
  and the drop in the maximum is largely due to no single standout game appearing

**Reason for not adopting**: 233ms → 856ms per move (**3.68x**) for +3.8%.
Teacher collection (`train_sim.py`) becoming 3.68x more expensive across the board is heavy. The effect itself is positive, so
**it is worth reconsidering if a cheaper lookahead can be written in the future** (not a refuted idea).

### Current penalty rules

`eval = score (merge points) - penalties (penalties for accidents and bad moves)`.
- per-move penalties in `_evaluate_drop`,
- board-wide penalties in `_board_penalties` (`src/policy.py`) are each added.

**Per-move penalties (`_evaluate_drop`)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| directly above a different type | `_foreign_aim_penalty` | when the fruit directly below the drop column (center offset within ±20%) is a different type | fixed 100.0 |
| blocking a waiting merge by burying | `_bury_block_penalty` | when a bigger fruit of another type blocks, directly above or on the shoulder, a fruit waiting for a same-type pair. **Only when the board is broken** (`board_is_broken`) | 14.0 ×type gap (half on a shoulder) |
| small-side escape after the floor fills | `_packed_small_side_penalty` | after the floor packs, when a large draw (orange or bigger) escapes to the small side (fires only when it physically cannot go on the small side) | fixed 8.0 |
| valley-growing bonus | `_valley_grow_ok` | landing in a valley whose fruit is the same type as held / whose fruit is one above held with held and next the same type. **Only when the board is broken** (`board_is_broken`) | **−3.0** (`VALLEY_GROW_BONUS`. The only bonus in this table) |

The 3 below apply **only when held itself did not merge** (`held_merged`, not the merge count
`merges`, so that an unrelated merge elsewhere on the board does not grant the exemption).

**Board-wide penalties (`_board_penalties`, on the post-drop board every time)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| dangerous height | inline | when the topmost crown is above danger_y(70.9) | (danger_y − crown) × 0.5 |
| burying | `_bury_penalty` | how much merge-candidate fruits are covered by other types (with sibling 1.0 / without 0.35) | bury_weight 20.0x |
| excess same type | `_excess_same_penalty` | 3 or more of the same type (up to 2 are allowed as waiting to merge) | 20.0 per excess fruit |
| size-order inversion | `_size_order_penalty` | pairs whose size order is inverted left to right (only fruits stuck in a valley of bigger fruits **and with a same-type partner left on the board** are exempt = `_size_order_exempt`. Valley fruits without a partner are counted). **Exempt on moves where held merged** (so unrelated fruits knocked by merge recoil are not counted as violations) | pair difference×1.5 + ideal_x deviation×0.004 |
| vertical size order | `_vertical_order_penalty` | for pairs overlapping horizontally and stacked vertically, when **the upper one is bigger**. Putting small fruits on shoulders (a ladder) has the smaller one on top, so 0 | type gap × 1.5 |
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
- `src/training/encode.py`: fixed-length observation vector
- `src/sim/sim_env.py`: headless drop sim (`sim_physics.simulate_drop`). `SimStep` is the real game's `score`
  only (no cumulative eval)
- Evaluation: `python scripts/eval_policy.py` (`--policy bootstrap|learned`. `--workers` default = logical cores/2)
- A/B: `python scripts/compare_policy.py`. Pits two bootstrap variants against each other, reporting not just means but
  per-seed wins and losses, per-phase metrics (`early_score` / `early_crown` / `dead_early` / `cascades`), and
  per-metric paired t values and 95% CIs. **Plug the change you want to compare into `_apply_variant`**
  (empty makes A and B identical, and a warning that every seed tied appears). A warning appears if even one `max_steps`
  A warning appears on truncation. Omitting `--seed` makes it random (reusing fixed seeds invites misreading).
  `--out` saves raw data as JSON
- Searching for proxy metrics: `python scripts/analyze_ab.py <dump.json>` (→[How to measure](#how-to-measure-traps-we-keep-stepping-in))
- Statistics are in `src/util/stats.py` (paired t / 95% CI / required n / correlation). scipy is not installed
- Training: `python scripts/train_sim.py` (collect → offline BC. The default max-steps=300 is a cap,
  not the losing line). best is score → moves → match
- Teacher collection runs in parallel with `ProcessPool` (default workers=logical cores/2; 8 on a 9700X; `--workers 1` for serial)
- `src/training/agent.py`: MLP with 32 discrete column bins / hidden 128 (old 20/64 npz files need retraining)
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

**Bug found (fixed)**: `MAX_FRUITS` in `src/training/encode.py` was 16,
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
and compares the results, and imitating from only the static features of `src/training/encode.py` (a list of fruit type/x/y/r)
a 1-hidden-layer MLP learning that "try candidates with physics, then choose" judgment
plateaued whichever of learning rate, epochs and soft/hard labels was tuned.
Unless the capacity (hidden width, layer count) or the feature design (such as including per-candidate landing results
in the features) itself changes, BC in this configuration is likely not to reach
the condition for starting RL. RL remains unstarted.
