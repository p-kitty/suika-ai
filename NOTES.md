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
Rewarding the rung count directly was measured and shelved
(→[ladder rung bonus](#tried-and-shelved-ladder-rung-bonus-2026-08-19)).

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

### Tried and shelved: ladder rung bonus (2026-08-19)

"Subtract built rung count × w from the penalties in `_evaluate_drop`" was implemented and put through
screening alone. **It was reverted without going to an A/B** (the implementation is not kept).

**The screening conditions were fixed before measuring**: (1) a move change rate of at least 3% at some w
(2) the distribution of rungs after landing shifts upward (3) in positions with rung 3+, the rate of taking merges
does not drop from w=0 (a watch for suppressing firing).

**Position set**: 428 positions taken every other move from move 60 onward, playing 6 seeds through.
The set first taken at moves 1-60 had 87% at rung 0 and only 4 cases of rung 3+,
seeing only boards before the mechanism engages. **A ladder is a shape that builds after the floor fills, so
an early-game position set measures nothing** (the first screening was wasted on this).

| w | move change rate | rungs after landing 0/1/2/3/4+ | moves taking a merge at rung 3+ |
|---|---|---|---|
| 0 | 0/428 (0.0%) | 296/36/28/27/41 | 3/66 |
| 5 | 5/428 (1.2%) | 294/37/29/26/42 | 3/66 |
| 10 | 6/428 (1.4%) | 294/36/29/27/42 | 2/66 |
| 20 | 7/428 (1.6%) | 293/37/28/27/43 | 2/66 |
| 40 | 12/428 (2.8%) | 291/37/30/26/44 | 2/66 |
| 80 | 17/428 (4.0%) | 290/37/30/24/47 | **0/66** |

**The w that bites and the w that does not break do not overlap.** Only w=80 satisfies (1), and there (3)
collapses completely. Firing a ladder drops the rungs from 4→0, so only firing moves carry −4w.
One ladder is 100 points, so even at w=20 the gain is effectively cut to +20, and at w=80 it cannot be taken.
**This term trades building a ladder against firing it, and the points are on the firing side**.

**The reason it fails is shape, not weight.** On the same 428 positions, counting "how far the rungs can be extended at most
among all candidates", **91.6% of positions are +0 rungs** (29 at +1, 7 at +2).
The rung bonus attaches to the post-drop board, so in positions with no candidate that can extend it every candidate takes
the same value and it vanishes as a constant difference. **The ceiling is 8.4%**, and w=20 actually moved 1.6%.
That the material cannot be drawn was already known (the "bottleneck is building it" above).
For the same reason that writing a rung as "a move that drops and places it" is a poor approach,
**rewarding "the rung count after the drop" hits the same wall**.

**score / cascades were not measured.** It was dropped at screening, so no A/B was run.
A 1.6% change in moves is buried in score noise at n=25.

**Reaching a watermelon in one game is no basis.** It started from the observation that seed 74546 reached
a watermelon at w=20, but when one move changes the board diverges completely from there, so
the stage reached in one game cannot be told apart from draw luck (→[How to measure](#how-to-measure-traps-we-keep-stepping-in)).

**If done next, change the shape.** Look not at the rung count itself but at "room to build a ladder"
(whether the base is on the wall with the inside open), or reward the side that draws the line of making rung material
by merging. Counting rungs after the drop has too few positions it can move.

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

### Screening called all three "good" (166-230 positions, deterministic)

**This section records "why it got adopted".** The A/B result is in the next section. It compares
candidates on the same position, a deterministic quantity with no noise, but **with no correspondence to score**.

| Variant | agreement | horizontal inversions | vertical inversions |
|---|---|---|---|
| vertical 1.5 / no gate | 89.8% | −0.24 | −1.06 |
| vertical 1.5 / gate 0.35 | 86.1% | −0.52 | −0.43 |
| vertical 3.0 / no gate | 88.0% | **+0.15** | −0.95 |

The trapped-fruit penalty at weights 0/2/4/8 gave total trapped −9/−18/−20/−27, inversions +34/+30/+26/+28,
merges 235/235/234/233. All read as "both trapping and inversions drop while merges stay the same".

The values decided from this (vertical 1.5, gate 0.35, trapping 4.0) all lost in the A/B.

**Facts picked up as a by-product** (they remain even though the rules were reverted):

- Counting the inversion rate only by the left and right of each pair is not enough. In the order orange, grape, apple,
  the grape is inverted only against the orange, giving just 1/3 = 0.333, but
  a fruit squeezed between two big fruits is clearly a broken shape. **A fruit in a valley should be counted as out of place
  with respect to both walls** (reusing `_valley_flanks` gives 2/3 = 0.667).
  This error was detected by `test_grows_valley_fruit_when_held_and_next_are_one_smaller`.
  **The test was right and the metric was wrong**
- Using only the presence of a valley to judge the board state does not work. With 8+ fruits there is always a valley, so
  it is always true. The inversion rate including valleys keeps a gradient by fruit count: 98% / 70% / 48% / 16% / 3%
  (fruits 0-3 / 4-7 / 8-11 / 12-15 / 16+)
- `_floor_packed` **reads 48% of boards with 3-5 fruits and 74% with 6-8 as "filled"**.
  The premise of `packed_small_side_penalty`, "the floor is packed and there is no room on the small side",
  does not hold from the early game (fixed in →[floor-filled check](#settled-floor-filled-is-judged-by-the-draw-2026-08-19))

### Settled: floor-filled is judged by the draw (2026-08-19)

The threshold of `_floor_packed` changed from a fixed orange diameter to the diameter of the draw.

`packed_small_side_penalty` is the only caller of `_floor_packed`, and through
`PACKED_BIG_DRAW_MIN_TYPE = SPAWN_MAX_TYPE - 1` **it runs only for 2 types,
dekopon and orange**. The old threshold was the fixed orange diameter of 77.1, so
the orange side was already correct and only dekopon (diameter 59.6) was off.
It read even gaps a dekopon falls into with 17.5 to spare as "no room".

- The floor-filled rate before the change (8 seeds × 40 moves) was **43.8%** at 3-5 fruits and **89.5%** at 6-8.
  A board merely lined up in a row (`pear@70 grape@216 straw@288 cherry@346`, largest gap 67.1)
  counted as filled for a dekopon draw
- Swapping the old and new predicates and running `choose_x` both ways on the same positions (6 seeds × 60 moves),
  **356/360 moves match (98.9%) with only 4 changed**. Penalty firings went 1445 → 1219 (**−15.6%**)
- **No A/B was run.** A 1.1% change in moves is buried in score noise at n=25
  (→[How to measure](#how-to-measure-traps-we-keep-stepping-in)). A change that fixes a wrong premise,
  making no claim of moving the score

### Tried and reverted: vertical size order, stage gate, trapped-fruit penalty (2026-08-18)

**All three lost their A/Bs and were reverted.** How they were made and why they were removed is kept.

**Trigger**: a report that view_sim showed a plainly dirty board. It collapsed into peach-orange-peach.

**The causation was this.** The collapse started 13 moves earlier:

- **Move 25**: the board is in perfect descending order, `peach@70 orange@172 straw@230 cherry@384`
  (inversion rate 0.000, trapped 0). When dropping a dekopon, the move trapping the strawberry beat
  the **13** candidates that do not trap it by **an eval difference of 0.07**. All 33 candidates sit on a tie plateau
  from −6.12 to −6.21, 0.09 wide
- **Move 28**: because of that trapping, the strawberry and orange swapped positions, and
  **every placement was already bad**. Even at ideal_x the local inversion amount was the same 3 units and could not be told apart
- **Move 38**: a 64-point 3-step cascade. Same result anywhere in x=204-276, and moves not taking the cascade
  raise trapping from 2→3. **Taking it is right**. The resulting peach appeared right of the orange and became fixed

**peach-orange-peach is the result, not the cause.**

**Wrong guesses (kept as a record)**:

- `packed_small_side_penalty` is the culprit → no. It does add 8.0 at move 28, but
  removing it does not change the choice. Tried on 241 positions, **100% agreement**; this rule does not affect
  move choice (though `_floor_packed` reading 48% of boards with 3-5 fruits as "filled"
  is a separate problem, which was fixed in
  [floor-filled check](#settled-floor-filled-is-judged-by-the-draw-2026-08-19))
- A local ordering term (counting per move the inversions the dropped fruit creates) works → at both move 25 and
  move 28, no change up to w=4.0. The fruit it inverts with is the same whether the landing is right or left, and
  **inversion counts cannot tell them apart**

**What was built (all reverted)**

| Rule | Content |
|---|---|
| vertical size order | for pairs overlapping horizontally and stacked vertically, type gap × 1.5 when the upper one is bigger |
| stage gate | apply the recovery rules (bury_block / valley growing) only on boards whose inversion rate exceeds a threshold |
| trapped-fruit penalty | 4.0 per fruit squeezed left and right by bigger fruits with no same-type partner left |

**A/B (n=25, same seeds, `--seed 526304`). All negative**:

| Variant | score | Δ | t | cascades | max_type |
|---|---|---|---|---|---|
| all three | 2045.60 | −5.5% | −1.04 | −7.9% | −1.3% |
| gate only | 2035.24 | −6.0% | −1.05 | −3.1% | −1.7% |
| vertical only | 1981.20 | **−8.5%** | −1.73 | −10.8% | — |
| trapping only | 2077.04 | −4.1% | −0.73 | −5.0% | −1.3% |

Side A was 2165.44 for all (with the 3 rules cut). None is significant (±100 points needs
n≈127), but **all 4 runs, every metric, are negative**. The worsened seeds all go max_type 10 → 9,
a clear drop in the metric closest to the double watermelon goal.

**The biggest lesson: home-made structural metrics pointed the wrong way all three times.**
Weights were chosen by screening on trapped and inversion counts, all three came out "good", yet score
dropped every time. Exactly the selection bias item in [How to measure](#how-to-measure-traps-we-keep-stepping-in).
**Do not use a proxy metric not validated against score as the basis for choosing weights.**
The only one allowed is the already validated `cascades`.

**Individual findings**:

- **The stage gate is logically unsound.** A valley is the shape "squeezed left and right by fruits bigger than itself",
  so **a big fruit on the small side is itself an ordering violation**. In other words
  *if there is a valley, there is always an inversion* (inversion rate > 0 on all 332 measured boards). So
  the state "we want to grow a valley but the board is tidy" does not exist, and the gate at threshold 0 is
  a pure no-op, while above 0 the only effect is the harm of **forbidding recovery on broken boards**.
  Measured, it stopped 28 (19%) of 151 positions where valley growing held,
  and those 28 had an inversion rate of at least 0.174
- **The premise of vertical size order is doubtful.** "47% of vertically stacked pairs are upside down = disorder = defect"
  was the reading, but since **merging creates the big fruit right there**, a big fruit sitting on small ones
  may be the normal state of this game. Horizontal inversions were traced to collapse on seed=74546,
  but vertically it was made a rule by analogy alone. At −8.5% alone it was the worst
- **The trapped-fruit penalty did fix the intended move** (the blunder at move 25 disappeared), yet still −4.1%.
  Counting fruits with partners too made it worse, dropping seed=74546 from 241 moves → 163
  (it was crushing the moves that feed a valley)

**Note**: all 4 used the same 25 seeds, so they are not independent evidence. It was not
confirmed on another seed set. But the full playthrough of seed=74546 independently pointed negative too.

### `bury_block` was retired (2026-08-18)

`bury_block_penalty` was deleted because **no effect could be detected with either structural metric**.
(At the time it was measured with vertical size order and the gate in place, which were later reverted, so
**this retirement alone remains unmeasured**. It needs a standalone A/B against master.)
Structurally too it is the type-gap penalty for "a big fruit on top of a small one"
(`14.0 ×(drop_type - under.type)` is itself the type-gap penalty for "a big fruit on top of a small one"),
with just a "the fruit below has a partner" condition and a 9x weight attached.

On 166 positions, removing it changes 10% of moves (agreement 89.8%). On top of that:

| Variant | total board dirt (horizontal + vertical inversions) | change in live merge pairs | merges |
|---|---|---|---|
| current | +40 | −2 | 168 |
| **retired** | **+40** | **±0** | 168 |
| only live pairs | — | −1 | 168 |

Dirt is exactly 0 net (the increases and decreases of the 17 changed moves cancel completely). **Even for its real job of
protecting merge pairs, removing it leaves slightly more.** The merge counts are identical for all 3.

The difference is too small, so the right reading is not "retiring is better" but "**no difference**".
If it has no effect, take the side that cuts it to one rule and can say plainly "while tidy, just line them up in order".
It also matches that 66% of what it protected was already unmergeable.

**Measured (n=25, seed=221700). No harm**. In a master worktree a variant zeroing
`bury_block_penalty` was set up and run with A=master / B=retired
(on the branch the whole function is gone, so it cannot be brought back from `_apply_variant`; measured the other way around):

| Metric | A → B | Δ |
|---|---|---|
| score | 1978.60 → 2014.88 | +1.8% (t=0.35) |
| steps | 207.3 → 210.0 | +1.3% |
| merges | 184.9 → 188.2 | +1.8% |
| cascades | 18.04 → 18.48 | +2.4% |
| max_type | 8.92 → 8.92 | ±0 |
| early_score | 225.44 → 227.60 | +1.0% |

Seed head-to-head win 13 / loss 12. No significant difference (±100 points needs n≈100), but **not one metric
leaned negative**. A contrast with the 3 rules measured the same day, "all 4 runs, every metric negative".

This is **not evidence of improvement but confirmation of no harm**. It is enough basis for removing one rule.

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

**Priority**: not breaking the board ranks above score. The final goal is not mean score but
a double watermelon, firing on the big side after waiting to draw one orange or two dekopons
(the shape `src/ladder.py` only detects). When deciding penalty weights, first check that the accident-avoidance side
is not overriding size order and trapping. The basis is
[how the board collapses](#investigated-how-the-board-collapses-and-isolating-the-stage-2026-08-18) and
[material arithmetic](#material-arithmetic-distance-to-a-double-watermelon).

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
| small-side escape after the floor fills | `_packed_small_side_penalty` | after the floor packs, when a large draw (dekopon, orange) escapes to the small side (fires only when it physically cannot go on the small side). Floor-filled is judged by the draw's diameter (→[floor-filled check](#settled-floor-filled-is-judged-by-the-draw-2026-08-19)) | fixed 8.0 |
| valley-growing bonus | `_valley_grow_ok` | landing in a valley whose fruit is the same type as held / whose fruit is one above held with held and next the same type | **−3.0** (`VALLEY_GROW_BONUS`. The only bonus in this table) |

The 3 below apply **only when held itself did not merge** (`held_merged`, not the merge count
`merges`, so that an unrelated merge elsewhere on the board does not grant the exemption).

**Board-wide penalties (`_board_penalties`, on the post-drop board every time)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| dangerous height | inline | when the topmost crown is above danger_y(70.9) | (danger_y − crown) × 0.5 |
| burying | `_bury_penalty` | how much merge-candidate fruits are covered by other types (with sibling 1.0 / without 0.35) | bury_weight 20.0x |
| excess same type | `_excess_same_penalty` | 3 or more of the same type (up to 2 are allowed as waiting to merge) | 20.0 per excess fruit |
| size-order inversion | `_size_order_penalty` | pairs whose size order is inverted left to right (only fruits stuck in a valley of bigger fruits **and with a same-type partner left on the board** are exempt = `_size_order_exempt`. Valley fruits without a partner are counted). **Exempt on moves where held merged** (so unrelated fruits knocked by merge recoil are not counted as violations) | pair difference×1.5 + ideal_x deviation×0.004 |
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
