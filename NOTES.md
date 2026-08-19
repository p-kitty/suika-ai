# Known issues and things to fix later

## Contents

- [Current approach: fixed-point observation of seed 642746](#current-approach-fixed-point-observation-of-seed-642746-2026-08-19)
- [Open tasks](#open-tasks)
- [How to measure](#how-to-measure-traps-we-keep-stepping-in) ← read before reporting numbers
- [Settled: the tie band really is indifferent](#settled-the-tie-band-really-is-indifferent-2026-08-19) ← the dead end of weight tuning
- [In progress: big draws and ladders after the floor fills](#in-progress-big-draws-and-ladders-after-the-floor-fills)
- [Investigated: how the board collapses](#investigated-how-the-board-collapses-2026-08-18)
- [Rules tried and reverted or retired](#rules-tried-and-reverted-or-retired)
- [Investigated: sudden death from scattered low-tier fruits late in the game](#investigated-sudden-death-from-scattered-low-tier-fruits-late-in-the-game)
- [Material arithmetic (distance to a double watermelon)](#material-arithmetic-distance-to-a-double-watermelon)
- [When to move](#when-to-move)
- [Policy (bootstrap) design](#policy-bootstrap-design)
- [Training](#training)
- [Planned: RL (REINFORCE)](#planned-rl-reinforce)

## Current approach: fixed-point observation of seed 642746 (2026-08-19)

**Look only at the one game of seed 642746 and fix obvious blunders one at a time. The goal is a corner watermelon
(and beyond that a double watermelon).**

The reason for this approach is in
[Settled: the tie band really is indifferent](#settled-the-tie-band-really-is-indifferent-2026-08-19).
The existing weights have no leverage, and no difference an A/B can pick up remains. Defects at the level of a single
position, on the other hand, are deterministic quantities, so they are visible without going through score noise.

The tracing procedure is in
[AGENTS.md](AGENTS.md#do-not-run-an-ab-while-obvious-blunders-remain).
Only the symptoms found and their diagnoses are kept here.

**Name moves by `move` in the `view_sim` footer (1-based).** Changing the policy makes play
diverge before that move, so the same move number points to the same position only before the change. To check whether it was fixed,
do not play through; replay the pre-change moves up to that position and compare there.

### Resolved: moves 52 and 45 (2026-08-19)

**Move 52 — knocking the pineapple out of the corner.** The cause was `packed_small_side_penalty`, resolved by
deleting it (→[Retired: packed_small_side_penalty](#retired-packed_small_side_penalty-2026-08-19)).

**Move 45 — burying a grape by putting an orange on it.** With packed removed, this one appeared instead.
The board is `peach@69 pear@192 dekopon@273 grape@329 cherry@384`, held=orange.

- The chosen x=312 makes the orange **sit on both the dekopon and the grape at once**
  (vertical gaps −14.2 / −7.4 = sinking into contact)
- **Yet `bury` is 0.00 for every candidate.** Against sideways offsets of 41.7 / 30.0, the window is
  only `under.radius * 0.9` = 26.8 / 23.8
- **The smaller the lower fruit, the narrower the window.** The shape we most want to crush, "burying a small fruit with a big one",
  escaped detection the more it was that shape: a definition working against its intent
- Fixing the window to `(under.radius + over.radius) * 0.9` gives 61.5 / 58.5 and detects both.
  x=312 drops out of the top 10 and it picks x=96, which buries nothing

**Comparison on 125 positions / 5 seeds:**

| Configuration | agreement with master | total buried fruits |
|---|---|---|
| master (with packed, old bury) | (baseline) | 15 |
| no packed, old bury | 95.2% | 18 |
| with packed, new bury | 78.4% | 12 |
| **no packed, new bury (adopted)** | **76.8%** | **13** |

- Fixing the bury window cuts burying by 20%, from 15 → 12. But **21.6% of moves change**
  (over 4x the 4.8% of deleting packed). A larger footprint than any existing intervention
- packed still slightly reduces burying even with the new bury (12 vs 13). Not a complete replacement
- **What looked like "0 buried" over one game of 80 moves was a chance board after divergence; 13 remain over 125 positions.**
  The lesson not to read structural metrics from a single playthrough was stepped on here too
- **Score was not measured.** "Total buried fruits" is a home-made structural metric not validated against score, and
  weights must not be chosen on its basis (→[How to measure](#how-to-measure-traps-we-keep-stepping-in))

### Reference: move 224 is a different kind despite the same "orange toward the pineapple"

The board is `apple@51 pineapple@78 melon@97 … pineapple@265 peach@331`. It chose x=72, but
**the top 12 candidates form a perfect tie band 0.0056 wide** (wherever it drops, it rolls and
merges with `orange@287`, so the real-game score ties at 15, and `packed`/`foreign_aim`/`bury` are
zero for every candidate, saturated. The only difference left is the ideal_x part of `_size_order_penalty`,
with 0.0013 between 1st and 2nd). **This was not chosen but merely picked from the band by order**, so
tuning weights does not fix it (→[Settled](#settled-the-tie-band-really-is-indifferent-2026-08-19)).
The same-looking blunder needs a different remedy from move 52, which a single term decides.

## Open tasks

**Vision**
- `10.png`: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than policy / held

**Physics simulation**
- Friction between fruits seems low: fruits slide in far more than in the real game.

**Training pipeline**
- Training episode length: raise `max_steps` and lower `episodes` (fewer, longer games).
  The default in `train_sim.py` is 300 (measured natural ends are median 210 moves and max 311, so
  about 2% are truncated. 320 for 0%)

## How to measure (traps we keep stepping in)

**Score noise is very large.** This is the biggest wall for improving the policy, and every attempt below
got buried here. Read this section before reporting numbers.

- The per-game standard deviation is ~1000-1200, and even in paired comparisons on the same seed the SD of the difference is 78-496.
  Seeing ±100 points as significant needs **n≈100**. Tens of episodes are not enough.
  It is faster to look for a proxy metric with lower variance than score
- **Do not judge by a rise or fall in the mean alone.** `compare_policy.py` prints, per metric, the paired t value and
  95% CI (`src/util/stats.py`). A row whose CI crosses 0 says nothing at that n.
  When not significant it also shows "the n needed to speak to ±100 points"
- **Divide out the required n before running.** `compare_policy.py` prints the SD every time, so
  the material is always at hand. Doubling the sample while saying "it is underpowered"
  and rerunning just makes the ruler bigger for a distance it cannot reach
- Add `--out artifacts/xxx.json` to long runs to keep per-seed raw data.
  It is written before aggregation, so a bug on the aggregation side does not lose hours of work
- **The proxy metric was settled as `cascades` (number of moves with 3+ merges) (2026-08-17).**
  It ranked near the top in all three independent dumps, with sensitivity ratios 1.34 / 1.28 / 1.40 and r(score) 0.83, stable.
  It reduces n only by the sensitivity ratio (at 1.4x the required n is about halved), so it is no silver bullet.
  Selection is done with `scripts/analyze_ab.py <dump>` (ranks metrics by n_detect)
- **Do not use a proxy metric not validated against score as the basis for choosing weights.**
  The 3 rules whose weights were chosen by screening with home-made structural metrics (trapped count, inversion count)
  all came out "good" in screening, yet every metric was negative in the A/B
  (→[rules tried and reverted](#rules-tried-and-reverted-or-retired)).
  Picking the metric that looked best in the same dump is also selection bias
- **The inversion rate is unusable as a per-episode metric (n=24).** The correlation between the all-move mean inversion rate and
  score is **+0.36** (the reverse sign, dirtier means higher score), confounded with game length.
  Removing the confound with a fixed number of moves gives r = 0.00-0.12, uncorrelated (compare: `steps` r=0.95,
  `cascades` r=0.76). **Use it only as a per-move difference, "how much did one move dirty the board"**
- **Discount the numbers when truncation happens.** Truncated games are the ones that went long, so
  the better the change the more it is underestimated. Natural ends over 200 measured runs: mean 213 / median 210 / max 311 moves.
  `--max-steps` **truncates 0% at 320 or more**; the default is 400. At 200 it is 64%,
  **at 100, 100% are truncated** (the 08-03 accident was this). `compare_policy.py`
  warns if even one is truncated
- **Do not stratify by the outcome and compare the same outcome (regression to the mean).** Splitting into top/bottom by A's score
  and comparing A with B always shows "the top got worse, the bottom improved".
  To compare distributions, compare quantiles directly
- **The value of a single move cannot be measured with rollouts.** Changing the move from a position and playing
  to the end, the board diverges completely from there, so draw luck does not cancel. Even pairing on the same draw sequence
  only lowers the SD of the difference from 580 → 449. **Even the lowest-eval blunder
  cannot be told apart from the chosen move** (400 pairs, Δ+31.8, t=1.37, win rate 49.8%).
  The required n is about 850 pairs for a 31.8-point difference, and **over 10,000 pairs** for the real 8.4-point difference.
  To see a difference, work **per policy** (`compare_policy.py`) instead of per position
- Do not fix seeds (omitting `--seed` makes them random). Reusing fixed seeds makes a chance collapse
  easy to misread as "reproduced". Compare changes paired on the same seeds

The measuring procedures themselves (how to plug in an A/B is in [AGENTS.md](AGENTS.md#when-touching-the-policy-or-training),
comparisons that do not dirty the working tree are in [git in AGENTS.md](AGENTS.md#git)) live there.
Only what can be trusted is written here.

## Settled: the tie band really is indifferent (2026-08-19)

**The most useful conclusion from the series of measurements on 08-19.** After four interventions to split the band all failed,
measuring the band itself settled it.

**A variant that deliberately shuffles the ranking inside the tie band randomly was measured at n=133**
(uniform noise in [0, 0.1) is added to held eval before sorting. The ranking outside the band does not change.
The noise is derived deterministically from the board, so it stays reproducible):

| Metric | A | B (band randomized) | Δ | t |
|---|---|---|---|---|
| score | 2111.60 | 2105.01 | −0.3% | −0.13 |
| steps | 218.4 | 217.7 | −0.3% | −0.18 |
| merges | 196.0 | 195.3 | −0.3% | −0.15 |
| cascades | 18.69 | 18.74 | +0.2% | +0.08 |
| max_type | 9.02 | 9.03 | +0.2% | +0.20 |

win/loss 68/65, paired difference −6.6 (SD of the difference 582). **Choosing at random inside the band loses
nothing is lost.** The current tie-break (the third decimal of bumpiness) does no work.

**The framing "the plateau is a defect" was wrong.** The candidates in the band really do lead to
equally good futures. The policy correctly recognized "these moves are equivalent";
it was not failing to choose.

### What the band actually looks like

Measured on 428 positions (move 60 onward, 6 seeds). Median 43 candidates.

- **The plateau is the norm.** Of 43 candidates, **9** are within eval 0.1 (11 at eps=0.5,
  14 at eps=2.0)
- **30% of the band is an artifact of candidate spacing, 70% is real.** Distinct landing outcomes in the band
  have a median of 2. In 31.8% of positions the band collapses to a single board (`CANDIDATE_STEP = 12.0` is
  simply finer than the physics resolution). The remaining **68.2% really contain 2 or more different boards, and
  22.0% contain 5 or more**
- **Inside the band every large term is saturated.** The width inside the band (eps=0.1) is
  bumpiness median 0.009 (moves in 44.8% of positions), big_layout 17.6%, danger 8.3%,
  size_order 2.1%, **bury / foreign_aim / packed 0.0%**
- **Zero width does not mean "dead".** Measuring absolute values over all candidates (60 positions / 2660 candidates),
  bury is nonzero for 41.9% of candidates with minimum −80.0, foreign_aim 18.6% with minimum −100.0,
  size_order 70.3% with minimum −200.1, excess_same 69.8% with minimum −280.0.
  **These terms do the job of "knocking bad candidates out of the band", and the candidates left in the band
  are already all tied on those terms.** All of them are discrete "count × weight" quantities, so they saturate

### Interventions that tried to split the band and failed

| Intervention | Moves change | Escapes the band | Result | Why it does not work |
|---|---|---|---|---|
| Ladder rung bonus | 4.0% (w=80) | — | dropped at screening | 91.6% of positions have no candidate that extends the rungs. At a w that bites, firing is lost |
| Trapped-fruit penalty | — | 1.2-4.7% | dropped at screening | only 1.2% of positions (eps=0.1) have the trapped count move inside the band |
| `drop_ideal` w=0.004 | 51.4% | **6.3%** | null at n=133 | every changed move stays inside the band |
| bumpiness x4 (0.08→0.32) | 17.5% | **7.0%** | −1.3% at n=133 | same. The band shrinks from 9→6 but score does not move |

- **Ladder rung bonus**: the screening conditions were fixed before measuring (1: move change rate at least 3%
  2: rung distribution after landing shifts upward 3: the rate of taking merges at rung 3+ does not drop). Only
  w=80 satisfies 1, and there 3 collapses from 3/66 → **0/66**. Firing a rung drops it from 4→0, so
  **only firing moves carry −4w**. This term trades building a ladder against firing it,
  and the points are on the firing side. The position set is 428 positions from move 60 onward on 6 seeds
  (the set first taken at moves 1-60 had only 4 cases of rung 3+, seeing only boards before the mechanism
  engages. **A ladder is a shape that builds up after the floor fills, so an early-game set measures nothing**)
- **Trapped-fruit penalty**: defined as "a fruit squeezed left and right by bigger fruits with no same-type partner left"
  (`_is_nestled`). Even loosening the band to eps=2.0, a difference appears in only 4.7%.
  The median eval difference needed to split the band is 0.004: "a tiny addition would catch it, but
  there are almost no positions to catch". This is also why it did not work in the 08-18 A/B
- **`drop_ideal`** (= `|land_x − ideal_x(drop_type, sign)| × 0.004`): it passed the screen
  (median spread 24.9 inside the band, width > 1 in 71.7% of positions). **It looked good at n=25 but
  vanished at n=133** (score +10.0% t=1.79 → +2.7% t=1.10 / cascades +20.3% t=2.78 →
  +4.7% t=1.58). At n=133 the only metric whose CI does not cross 0 is `early_crown`, and it **favors A over B**.
  win/loss 64/69, median paired difference −8. The 64 wins average +526 against −381 for the 69 losses:
  **a shape that "widens the swing" rather than "makes it better"**.
  *It is worth recording that cascades, significant at n=25, vanished at n=133. A proxy metric only reduces the required n
  by a factor of 1.4, and does not justify n=25.*
  *Replaying one lost game (seed=982108) seemed to show the big-fruit cluster scattering, but it diverged on move 2,
  and pairing on the same 428 positions shows no difference in cluster spread (Δ +0.05, t=0.66).
  An example of how reading structural causation from a single trace goes wrong.*
- **bumpiness x4**: the minimal intervention of raising the weight of the only term that routinely moves inside the band.
  Multiplier and move change rate: x2 8.4% / x4 17.5% / x8 23.4% / x16 32.2%, and the band size
  shrinks 9 → 8 / 6 / 4 / 3. At n=133 score −1.3% (t=−0.49), CI [−135.0, +81.7].
  **There is no reason to move the existing 0.08**

**A wide band is a symptom, not the disease.** Making the tie-break deterministic with an arbitrary continuous quantity
just chooses in a different arbitrary way. The screen (does that quantity vary inside the band) is
**necessary, not sufficient**. Without traction it certainly will not work
(ladder, trapped), but with traction it does not necessarily work (ideal_x, bumpiness x4).

### Screen on "does it escape the band"

Since the inside of the band is indifferent, **"what fraction of moves change" is not a screen**. What to look at is
**the fraction of chosen moves that leave the original band**. `python scripts/band_escape.py`
(one physics pass + an analytic sweep, minutes). `drop_ideal` 50.2% → 6.3% and
bumpiness x4 17.5% → 7.0% in the table above verify its predictive power.

### The existing weights have no leverage

What decides which candidates enter the band is the large terms, so their weights were swept
(428 positions, `--eps 0.1`). **Fraction escaping the band:**

| Term | x0.5 | x1.5 | x2.0 |
|---|---|---|---|
| bury | 0.9% | 0.9% | 1.6% |
| excess_same | 0.5% | 1.2% | 1.9% |
| size_order | **3.0%** | 1.2% | 1.9% |
| big_layout | 0.5% | 0.2% | 0.2% |
| foreign_aim | **0.0%** | **0.0%** | **0.0%** |

It reproduces with every term at 2.8% or less on 3 independent seeds (176 positions).

- **The maximum is 3.0% from halving size_order.** Even `drop_ideal`, which changed 50% of moves, was
  null at n=133, so measuring 3% finds nothing
- **`foreign_aim` does not change a single move at 0.5x or 2x.** The weight 100.0 is so large that
  candidates it applies to are out of contention from the start. Effectively a binary "applies or not" filter
- **`big_layout` changes 5.1% of moves but only 0.2% escape the band**

**Conclusion: bootstrap is somewhere weight tuning cannot get out of.**
The inside of the band is indifferent, the weights deciding who enters the band have no leverage, new count terms do not move
inside the band, new continuous terms only move inside it, and the road of a deeper search was
[measured and shelved on 08-17](#run-cost-faster-physics-and-search-width-2026-08-17).
To go further it is either **the definition on the side that decides who enters the band** (not weights, but what counts as a penalty), or
an evaluator that sees differences the current features cannot (a learned value function).
This matches how [Policy (bootstrap) design](#policy-bootstrap-design) has positioned it from the start:
"a thin policy before RL".

## In progress: big draws and ladders after the floor fills

After the floor fills, placing big draws such as orange / dekopon on the small side crushes the fruits below and
the board collapses. **`packed_small_side_penalty` was retired on 2026-08-19**
(→[Retired: packed_small_side_penalty](#retired-packed_small_side_penalty-2026-08-19)).
What stops this harm now is `_bury_penalty`.

**The ladder** (a pear next to the inside of a corner peach, an apple and an orange on the **shoulders** of those two, and the final orange
firing a 4→5→6→7 cascade) is a shape that arises naturally as a result of this placement rule.
`src/ladder.py` has only detection written, and **has never been called from the production path**
(only `tests/test_policy.py` calls it). It is kept as groundwork for using it in move selection.

- **Firing needs no guidance**. Once a ladder is built, `choose_x` ties with the best of an exhaustive
  sweep over x. Adding candidate x for firing was a no-op (removed)
- **`FOREIGN_AIM` is unrelated to the ladder**. The rungs sit on shoulders, so this penalty does not apply
- **`_size_order_penalty` is not in the way either**. Measured on ladder boards it is only 0.14-0.49
- **Without a filled floor the shape does not hold**. The pear is pushed out like a wedge and self-destructs, and wherever you drop
  you get only one rung (15 points). Filling the floor to the right edge gives 100 points. A filled floor is a gate condition
- **The bottleneck is building it**. A board with all 4 rungs appears only 12 times in 720. Writing a rung as a condition of "a move
  that drops and places it" is a poor approach (draws go up to orange; pear and apple can only be made by merging).
  Rewarding the rung count directly was also measured and shelved
  (→[Interventions that tried to split the band and failed](#interventions-that-tried-to-split-the-band-and-failed))
- The small-side room check confirms with an actual `simulate_drop`, not just the geometric gap width
  (`_small_side_room_ok`). Fixed a bug that judged a gap blocked by a roof as having room

## Investigated: how the board collapses (2026-08-18)

6 games were traced and counted per move (deterministic per-position quantities, so not
subject to score noise). The policy of "not breaking the board matters more than score" was reversed
in the weighting of the penalties.

**Symptom**

- **The inversion rate of horizontal size order goes from 10% early to 40-50% late**. 50% is complete disorder
- **Vertically it was lawless from the start**. Of 23079 vertically stacked pairs, 47% have "the upper one bigger".
  `_size_order_penalty` only looked at `a.x <= b.x`, and there was not a single vertical rule
- cherry / strawberry are the main culprits (+1.25 / +1.57 pairs per move). Only orange recovers, at −0.93
- **Clean moves are among the candidates. The evaluation rejects them**: for cherry, a non-dirtying move is a candidate
  in 97% of positions, yet it is actually chosen in 64%. 2.85 pairs missed per move

**What was beating size order** (in positions where a clean move was rejected, chosen move − clean move)

| Term | difference | rate of being the deciding factor |
|---|---|---|
| `bury_block` | **−7.43** | 35% |
| `FOREIGN_AIM` | −6.12 | 6% |
| valley growing | −1.04 | 35% |
| `sizeord` | −0.03 | 16% |

Converted, one inversion pair ≈ eval 4.91, whereas `bury_block` is 14.0 at type gap 1 and
`FOREIGN_AIM` is 100.0. **`sizeord` effectively does not distinguish dirty moves from clean ones.**

- `bury_block` fires on 51-72% of small-side candidates, and **66% of the pairs it protects already
  have a bigger fruit wedged between them and cannot merge**. 83% are more than 3x the contact distance apart
- Valley growing: **100% of its 1642 firings land on the big side**. 97% are "a same-type fruit is in the valley but
  this move does not merge", that is, moves that stack next to that fruit
- The `_size_order_exempt` exemption is an accomplice. Removing every exemption only takes it from 4.91 → 6.20 per pair

**Facts picked up as a by-product** (they remain even though the rules were reverted):

- Counting the inversion rate only by the left and right of each pair is not enough. **A fruit in a valley should be counted as
  out of place with respect to both walls** (reusing `_valley_flanks`). This error was
  detected by `test_grows_valley_fruit_when_held_and_next_are_one_smaller`.
  **The test was right and the metric was wrong**
- Using only the presence of a valley to judge the board state does not work. With 8+ fruits there is always a valley, so it is always true.
  The inversion rate including valleys, by fruit count, is 98% / 70% / 48% / 16% / 3% (fruits 0-3 / 4-7 / 8-11 / 12-15 / 16+)
  and keeps a gradient

**Settled: floor-filled is judged by the draw (2026-08-19)**

The threshold of `_floor_packed` changed from a fixed orange diameter to the diameter of the draw.
`packed_small_side_penalty` is its only caller, and through `PACKED_BIG_DRAW_MIN_TYPE`
**it runs only for dekopon and orange**. The old threshold was the fixed orange diameter of 77.1, so
the orange side was already correct and only dekopon (diameter 59.6) was off.

- The floor-filled rate before the change (8 seeds × 40 moves) was **43.8%** at 3-5 fruits and **89.5%** at 6-8.
  A board merely lined up in a row counted as filled for a dekopon draw
- Swapping old and new and running `choose_x` both ways on the same positions (6 seeds × 60 moves),
  **356/360 moves match (98.9%)**. Penalty firings went 1445 → 1219 (**−15.6%**)
- **No A/B was run.** A 1.1% change in moves is buried in score noise at n=25.
  A change that fixes a wrong premise; it makes no claim of moving the score

## Rules tried and reverted or retired

### Vertical size order, stage gate, trapped-fruit penalty (2026-08-18, reverted)

**Trigger**: a report that view_sim showed a board collapsed into peach-orange-peach.

**The cause was 13 moves earlier.** At move 25 the board was in perfect descending order (inversion rate 0.000, trapped 0),
and when dropping a dekopon **the move that traps the strawberry beat the 13 that do not trap it by an eval difference of 0.07**
(all 33 candidates on a tie plateau 0.09 wide). By move 28 every placement was already bad.
Taking the 64-point 3-step cascade at move 38 was right, and the resulting peach became fixed.
**peach-orange-peach is the result, not the cause.**

**What was built (all reverted) and the A/B (n=25, same seeds `--seed 526304`)**:

| Variant | Content | score | Δ | cascades |
|---|---|---|---|---|
| vertical size order | type gap × 1.5 when the upper of a vertical stack is bigger | 1981.20 | **−8.5%** | −10.8% |
| stage gate | apply recovery rules only on boards whose inversion rate exceeds a threshold | 2035.24 | −6.0% | −3.1% |
| trapped-fruit penalty | 4.0 per fruit squeezed left and right by big fruits with no partner | 2077.04 | −4.1% | −5.0% |
| all three | | 2045.60 | −5.5% | −7.9% |

Side A was 2165.44 for all. None is significant (±100 points needs n≈127), but
**all 4 runs, every metric, are negative**. The worsened seeds all go max_type 10 → 9,
a clear drop in the metric closest to the double watermelon goal.

- **The stage gate is logically unsound.** A valley is the shape "squeezed left and right by fruits bigger than itself",
  so **a big fruit on the small side is itself an ordering violation**. *If there is a valley, there is
  always an inversion* (inversion rate > 0 on all 332 measured boards). A threshold of 0 is a pure no-op,
  and above 0 the only effect is the harm of **forbidding recovery on broken boards** (it stopped 28 of
  151 positions where valley growing held)
- **The premise of vertical size order is doubtful.** Since **merging creates the big fruit right there**,
  a big fruit sitting on small ones may be the normal state of this game.
  Horizontally it was traced to collapse on seed=74546, but vertically it was made a rule by analogy alone
- **The trapped-fruit penalty did fix the intended move** (the blunder at move 25 disappeared), yet still −4.1%

**A wrong guess**: `packed_small_side_penalty` is the culprit → no. Tried on 241 positions,
**100% agreement**; this rule did not affect move choice. A local ordering term too
showed no change when tuned up to w=4.0 (the fruit it inverts with is the same whether the landing is right or left).

**Note**: all 4 used the same 25 seeds, so they are not independent evidence. But the full playthrough of seed=74546
independently pointed negative too.

**The biggest lesson**: home-made structural metrics pointed the wrong way all three times
(→[How to measure](#how-to-measure-traps-we-keep-stepping-in)).

### Retired: `packed_small_side_penalty` (2026-08-19)

**It was the only penalty that decided firing from pre-drop geometry alone.** It predicted before the drop the floor gaps (`_floor_packed`) and
whether the fruit could go on the small side (`_small_side_room_ok`), and imposed a fixed 8.0 when hit.
A rule measured at n=100 on 2026-08-06 and **made permanent as ON** (score 2047.0 → 2093.9, **+46.9**,
t=0.85, 95% CI [−62.3, +156.1], **not significant**. Seed head-to-head win 52 / loss 38 / tie 10.
Significance would need n=529 = about 14 hours, so it was adopted because the point estimate was positive, the shape matched the intent and it fired 90/100).
**That decision was overturned and it was deleted.**

**Basis for deletion** (125 positions / 5 seeds, deterministic per-position quantities):

| | |
|---|---|
| candidate-based firing | 189/5146 (3.7%) |
| position-based firing | 16/125 (12.8%). only dekopon and orange draws |
| **applies to the chosen move** | **0/125 (0.0%)** |
| removing it changes the move | 6/125 (4.8%). 6/16 (37.5%) restricted to positions where it fired |
| **penalized candidates whose post-drop board is better than the "chosen move"** | **75/189 (39.7%)** |

- **The rule only works on the side of excluding candidates** (it never applies to the chosen move)
- **40% of penalized candidates have a better post-drop board than the move actually chosen.** In other words
  it points the opposite way from `board_penalties`
- `_big_layout_penalty` **never agrees** with packed. Of 189 candidates, 0 point the same way,
  132 (69.8%) have the same value, and 57 (30.2%, median −0.429) point the opposite way. It is an order of magnitude
  smaller than the weight 8.0 and cannot compete

**Concrete example: move 52 of seed 642746** (`move` in `view_sim`). An orange draw, sign=+1.
The largest floor gap 73.2 < orange diameter 77.1, so `_floor_packed` is true and `_small_side_room_ok`
is false. 8.0 is added to every small-side candidate. Splitting the board penalties per term:

| Term | x=60 (the chosen move) | x=216 | difference |
|---|---|---|---|
| size_order | 6.213 | 0.055 | **+6.159** |
| big_layout | 0.000 | 0.000 | 0.000 |
| others | — | — | about 0 |
| packed | 0.000 | **8.000** | **−8.000** |
| total | 9.749 | 11.592 | −1.843 |

**`_size_order_penalty` was correctly charging 6.159 more. The fixed 8.0 of packed
overrode it and let the blunder win.** As a result it jammed an orange of diameter 77.1 into the 28.9 gap between the left wall and the pineapple,
and **the pineapple is knocked 102.6 from the corner to the center of the board** (gap from the wall
28.9 → 131.5). With packed cut it picks x=216, which instead pushes the pineapple toward the wall (28.9 → 3.3).

**No A/B was run.** Since it overturns a decision to make something permanent, it should have been measured on the same footing (n=100),
but the adoption A/B itself was not significant (its CI crosses 0), and the per-position evidence above is decisive, so
it was deleted first. **Rechecking by score has not been done.**

### `bury_block` was retired (2026-08-18)

`bury_block_penalty` was deleted because **no effect could be detected with either structural metric**.
Structurally, `14.0 × (drop_type - under.type)` is itself the type-gap penalty for "a big fruit on top of a small one",
with just a "the fruit below has a partner" condition and a 9x weight attached.

On 166 positions, removing it changes 10% of moves. Even so, board dirt (horizontal + vertical inversions) is +40 current /
+40 retired, exactly 0 net, and **even for its real job of protecting merge pairs, removing it leaves slightly
more** (−2 → ±0). The merge counts are identical.

Measured at n=25 (seed=221700) with A=master / B=retired, score 1978.60 → 2014.88 (+1.8%,
t=0.35), steps +1.3%, merges +1.8%, cascades +2.4%, max_type ±0. win 13 / loss 12.
No significant difference, but **not one metric leaned negative**. This is **not evidence of improvement but
confirmation of no harm**. It is enough basis for removing one rule.

**Note**: at the time it was measured with vertical size order and the gate in place, which were later reverted,
so **this retirement alone remains unmeasured**.

### Ideas that did not work (dropped at screening)

- **`size_order_pair_weight` from 1.5 → 9.0**: agreement 88.6%, and the inversion increase of changed moves is **+0.11**.
  It is a global statistic, the pair count of the whole board, so dropping one fruit is buried in the baseline. **This line is dead**
- In contrast, a local term counting per move "the inversions the dropped fruit itself creates" has agreement 80.7% at w=1.0,
  and changed moves average **−4.00 pairs**. The difference was local versus global
- **Running BC on only the top teacher data**: won't do. It cannot even fit the unfiltered teacher
  (→[BC does not reach 60-70% match](#investigated-bc-does-not-reach-60-70-match-2026-08-05)),
  and **`choose_x` is deterministic, so the spread of score between seeds is 100% draw-order luck**
  (n=24, mean 2012, SD 332). Picking the top means learning "how it played when lucky".
  If done at all the order is reversed: first make `src/training/encode.py` candidate-conditioned

## Investigated: sudden death from scattered low-tier fruits late in the game

**Diagnosing the cause of death**: every episode ends in `dead` and never reaches the `max_steps` cap.
Replaying one episode move by move (seed=20260816), for 10+ moves before death
low-tier fruits such as 5-8 cherries, 3 grapes and 4 dekopons remained scattered and unmerged.
The final move had no safe landing and died instantly. In the position just before it, dropping a cherry anywhere from x=0-400
produces no merge at all. So it is not a problem with "that move": much earlier,
low-tier fruits were squeezed from the sides by big fruits of other types and **scattered into physically unmergeable positions**,
which is the root cause.

**Improvement attempts (none confirmed; code reverted)**

| Attempt | Content | Result |
|---|---|---|
| triangular excess-same | `_excess_same_penalty` from linear to triangular | splits: +1.9% at n=24 / -3.0% at n=40 |
| side isolation penalty | detect moves that block both sides of a partnerless fruit | -9.2% at n=32 (difference 195 against SE~206) |
| three-ply lookahead | expand the third move for only the top 2 by eval. 313→401ms per move | -0.7% at n=32 |

### Investigated: the bonus side of eval is not the bottleneck (2026-08-17)

**Hypothesis (wrong)**: since the bonus is the real game's score itself, a cherry merge is worth only 1 point, while
the penalties are on the order of `EXCESS_SAME` 20 and `FOREIGN_AIM` 100. So perhaps the motive to clean up small fruits is
structurally buried.

**What was tried**: `FRAGMENT_WEIGHT`, penalizing the number of fruits on the board itself. One merge lowers the count
by 1, so it is mathematically the same shape as "a merge bonus independent of type".

**It failed before reaching an A/B.** In agreement screening (180 positions) the fraction choosing the same x as current is
**98.9-98.3% at w=3-15, and still 94.4% at w=25**.

**Why it does not work** (270 moves recounted): moves that merged 138 (51%), **moves that passed up
an available merging x 6 (2%)**, moves with no merge available anywhere 126 (47%).
**In 96% of positions where a merge is possible, the current policy already takes it.** Even with a 1-point bonus, merging
removes a fruit and lifts the bury, excess-same and height penalties wholesale, so
**lifting penalties was standing in for the bonus**.

**The view of the cause of death is updated too.** Measured by game progress (3 games, 575 moves), the fraction of "moves that can merge" is
47% early → 57% late, and **does not fall late**. Meanwhile the fruit count keeps rising, 4.3 → 17.0.
The late collapse is not "**it can no longer merge**" but "**merging cannot keep up with supply**".
Each move always adds one fruit, while a single merge removes only one; only cascades remove several.
This is also consistent with `cascades` consistently being the sharpest proxy metric.

## Material arithmetic (distance to a double watermelon)

Spawns are uniform over type0-4, so 6.2 cherry units per move. One watermelon = 1024 units,
a double watermelon = 2048 units. Measured (average of 6 games): 1217 units on the board at 192 moves, 59% of what is needed.

But **by area there is enough**. The area needed to hold the same 1024 units is 39k for one watermelon,
57k for two melons, 299k for 64 oranges, 651k for 1024 cherries (**17x less when consolidated**).
The board holds 135k measured at death, so holding 2048 units as two watermelons (79k)
physically fits. **What is missing is neither material nor moves, only consolidation.**

## When to move

- Do not decide x on a moving board. Waiting for it to settle takes priority over lookahead (`src/game/settle.py`)
- Wait for creep not only on instantaneous velocity but also on sideways drift while quiet

## Policy (bootstrap) design

**Priority**: not breaking the board ranks above score. The final goal is not mean score but
a double watermelon, firing on the big side after waiting to draw one orange or two dekopons
(the shape `src/ladder.py` only detects). When deciding penalty weights, first check that the accident-avoidance side
is not overriding size order and trapping. The basis is
[how the board collapses](#investigated-how-the-board-collapses-2026-08-18) and
[material arithmetic](#material-arithmetic-distance-to-a-double-watermelon).

- `src/policy.py` is a thin policy before RL. Only merging, dangerous height, burying, light size order and
  accident prevention for rolling / knock-aways
- The physics of falling, collision and merging is pymunk (`src/sim/sim_physics.py`; UT in `tests/sim/test_sim_physics.py`).
  `choose_x` scores with the same `simulate_drop`
- Moves are scored as `eval = score - penalties`. The only bonus is the real game's score; dangerous height, accidents and burying are penalties
- next lookahead: only the top `HELD_TOP` by held eval are re-evaluated with candidates at spacing `NEXT_CANDIDATE_STEP`
  (multiplied by `NEXT_DISCOUNT`). The physics is heavy, so it is coarser than held
- Burying is the main penalty. Moves that block a same-type pair waiting to merge with a bigger fruit of another type, directly above or on the shoulder, are heavily penalized
- Aiming at the center of a different type (`FOREIGN_AIM`): penalized if the fruit directly below is a different type and in its center band. OK if it is the same type.
  Not cut by `merges` (closes the loophole of rolling off a different type below and merging). Stacking a different type in valleys or on shoulders
  is not itself forbidden
- Excess same type (`EXCESS_SAME`): once 3 or more of a type accumulate, a penalty of 20 per excess fruit
- Valley growing (`valley_grow_ok`): the valley fruit is the same type as held, or the valley fruit is one above held and
  held and next are the same type: a bonus for landing in that valley. The reference is the valley fruit; the wall types are not looked at
- Layout: big fruits stay close together. On the big side (`sign`), the corner pocket outside an edge-anchored L and below L's center
  is heavily penalized (`_big_layout_penalty`)
- But if a type is missing between two neighbors, that much gap is not closed. Closing it leaves
  no place for the missing type when it is drawn, and the only option is to send it outside and break the order
- Not included: push-in merges, restoring pushes, cascade gap opening, forced moves one tier up, hard-coded ladder firing
- Do not add UTs for concrete procedures. When something breaks, look at accident prevention or the observation side
- Do not make `CANDIDATE_STEP` coarser. At 20 the spot directly above a dangerous pile lands on the grid and
  `test_avoids_dangerous_tall_stack` fails. Speed is earned on the lookahead side
- Cutting `SLEEP_FRAMES` does not work. A single `choose_x` gets faster, but the board settles differently and
  later moves get heavier, so the whole episode is actually slower (measured at 25). The physics fidelity
  (shared with `SimEnv`) also drops

### Current penalty rules

`eval = score (merge points) - penalties (penalties for accidents and bad moves)`. Per-move penalties are added in
`_evaluate_drop` (`src/policy.py`), board-wide penalties in `board_penalties`
(`src/penalties.py`).

**Per-move penalties (`_evaluate_drop`)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| directly above a different type | `foreign_aim_penalty` | when the fruit directly below the drop column (center offset within ±20%) is a different type | fixed 100.0 |
| valley-growing bonus | `valley_grow_ok` | landing in a valley whose fruit is the same type as held / whose fruit is one above held with held and next the same type | **−3.0** (the only bonus in this table) |

The valley-growing bonus applies **only when held itself did not merge** (`held_merged`, not the merge count
`merges`, so that an unrelated merge elsewhere on the board does not grant the exemption).

**Board-wide penalties (`board_penalties`, on the post-drop board every time)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| dangerous height | inline | when the topmost crown is above `DANGER_Y`(70.9) | (DANGER_Y − crown) × `DANGER_CROWN_WEIGHT` 0.5 |
| burying | `_bury_penalty` | how much merge-candidate fruits are covered by other types (with sibling 1.0 / without 0.35). The contact window is based on **both radii** `(under.radius + over.radius) × 0.9` (based on the lower fruit alone, the window narrows the more a big fruit sits on a small one and it escapes detection) | `BURY_WEIGHT` 20.0x |
| excess same type | `_excess_same_penalty` | 3 or more of the same type (up to 2 are allowed as waiting to merge) | 20.0 per excess fruit |
| size-order inversion | `_size_order_penalty` | pairs whose size order is inverted left to right (only fruits stuck in a valley of bigger fruits **and with a same-type partner left on the board** are exempt = `_size_order_exempt`). **Exempt on moves where held merged** | pair difference×1.5 + ideal_x deviation×0.004 |
| big-fruit layout | `_big_layout_penalty` | (1) the biggest fruit is on the big-side wall yet a small fruit is outside and below it (corner pocket filled) (2) big fruits not close enough (exempt for the diameter of the missing type in between) | (1) 50.0×(1+0.05×type gap)+depth×0.15  (2) (gap−diameter of the missing type)×0.025×size factor |
| bumpiness (height variance) | `_height_variance` | spread of crown heights per column bin (scaled by `VARIANCE_DANGER_SCALE` 0.15 at dangerous height) | variance× `VARIANCE_WEIGHT` 0.08 |

Notes:
- The rules above have no ON/OFF toggles (the policy is not to keep toggles for permanent rules.
  The A/B procedure is in [AGENTS.md](AGENTS.md#when-touching-the-policy-or-training))
- Ladder detection (`src/ladder.py`) is currently unused by penalties (detection only)

### Run cost: faster physics and search width (2026-08-17)

The search cost is essentially the number of `simulate_drop` calls. `HELD_TOP` / `NEXT_CANDIDATE_STEP`
decide the run time.

**Faster physics: the unit cost went from 9.42ms → 3.86ms (2.44x).** Play is completely unchanged, so
score was not measured. 99.2% of `choose_x` is `simulate_drop` (72 calls per move), and
**57.9% of that was `_find_merge_pair`**. On a 10-fruit board it was called 55,104 times per drop
and **returned None 100% of the time**. `_MergeScan` (skipping the scan by computing from the gap of the nearest same-type pair and the max speed
"how many more substeps contact is impossible") brought it from 55,104 → 3,717 calls
for 4.43ms, and reading `_all_quiet` from the back and similar changes gave 3.86ms.

- The lower bound on the time for a gap to close comes from the positive root of `g*T² + 2vT = gap`, since speed only increases through gravity
  (`elasticity=0`). Push-out corrections do not show up in `body.velocity`, so
  a 60px/s margin (`SCAN_SPEED_MARGIN`) on the speed and a cap of 16 substeps are applied
- **How it was verified (for this kind of change, score is not measured)**: record the x / y of every fruit after each move
  with `repr()`, together with the chosen x, score and merge count, and **compare byte for byte** with the output of a master
  worktree. 3 seeds × 70 moves and 2 seeds × 170 moves all matched. If the skip is too long by even
  one substep, a merge shifts and every later trajectory changes, so it is a sensitive check
- Little headroom remains (physics ~62% / quiet gate 18.5% / scan 10.8% / setup 5.9%)

**Search width 8/16 was remeasured and not adopted.** Agreement screening on 210 positions gave
75.2% agreement (1 move in 4 differs), so it went to an A/B. At n=100 (2.4 hours, 0 truncated)
score 2105.7 → 2185.3 (**+79.6 / +3.8%**, t=1.68, CI [-14.3, +173.5]). **Not significant**, but
**all 9 metrics lean toward B**, and only `cascades` +6.6% is significant (t=2.36).
Seed head-to-head win 59 / loss 41. Significance would need n≈136.

- The point estimate of +3.8% **nearly matches, in an independent measurement,** the old record of "−3.4% for 2/32"
- A direct quantile comparison shows it **lifts the bottom and trims the top** (min 1148→1499,
  median 2076→2179, max 3489→3215). But type10 reached rose from 12→20 runs
- **Reason for not adopting**: 233ms → 856ms per move (**3.68x**) for +3.8%.
  Teacher collection (`train_sim.py`) becomes 3.68x more expensive across the board. **It is not a refuted idea**, so
  it is worth reconsidering if a cheaper lookahead can be written
- **Follow this order (minutes of screening → hours of A/B) from now on too**

## Training

- **score / penalties**: `score` is the real game's merge score (1-65, no penalties), `penalties` are the penalties for accidents and bad moves.
  `eval = score - penalties` is **only for bootstrap move selection** (`choose_x`). The student's quality, saving and
  logs use the real game's `score` (moves on ties). **The RL reward is score too** (dense penalties are not rewards)
- `src/reward.py`: `merge_score(merge_types)` gives only merge points identical to the real game (cherry→0 …
  watermelon 55, double clear 65). No survival bonus or death penalty
- `src/training/encode.py`: fixed-length observation vector
- `src/sim/sim_env.py`: headless drop sim. `SimStep` is the real game's `score` only
- Evaluation: `python scripts/eval_policy.py` (`--policy bootstrap|learned`. `--workers` default = logical cores/2)
- A/B: `python scripts/compare_policy.py`. **Plug the change you want to compare into `_apply_variant`**
  (empty makes A and B identical, and a warning that every seed tied appears). A warning appears if even one `max_steps`
  truncation occurs. Omitting `--seed` makes it random. `--out` saves raw data as JSON
- Searching for proxy metrics: `python scripts/analyze_ab.py <dump.json>`
- Screening: `python scripts/band_escape.py` (→[Screen on "does it escape the band"](#screen-on-does-it-escape-the-band))
- Statistics are in `src/util/stats.py` (paired t / 95% CI / required n / correlation). scipy is not installed
- Training: `python scripts/train_sim.py` (collect → offline BC. The default max-steps=300 is a cap,
  not the losing line). best is score → moves → match
- Teacher collection runs in parallel with `ProcessPool` (default workers=logical cores/2; 8 on a 9700X; `--workers 1` for serial)
- `src/training/agent.py`: MLP with 32 discrete column bins / hidden 128
- Live play: `python main.py` (defaults to learned if an npz exists. `L` toggles bootstrap, `--policy bootstrap`)

### Investigated: BC does not reach 60-70% match (2026-08-05)

Measuring the existing checkpoints, score was only about half of bootstrap (~2000-2150)
(~1040-1060), far from the condition for starting RL.

**Bug found (fixed)**: `MAX_FRUITS` in `src/training/encode.py` was 16, packing the board's fruits
starting from the biggest types and cutting off the rest. Late in the game boards with over 20 fruits are common
(23 measured), and the first to be cut were cherry/strawberry —
the very culprits of the accident identified in [sudden death from scattered low tiers late](#investigated-sudden-death-from-scattered-low-tier-fruits-late-in-the-game)
were invisible to the student. Raised to 32 so effectively nothing is cut.

**It still did not solve it**: with the teacher data fixed (150 ep, n=32562),
an exhaustive sweep of lr 0.05-1.0 × epoch 80-300 × hidden 128/256 did not reach match 30%
(best: 29.1% with hard label, lr=0.5, hidden=256, epoch=300), and score plateaued at 900-1150.
The soft label approach was actually lower (15-21%).

**What we learned**: bootstrap is a policy that actually runs `simulate_drop` per candidate and compares the results,
and a 1-hidden-layer MLP imitating that judgment from only the static features of `encode.py` (a list of fruit type/x/y/r)
plateaued whichever of learning rate, epochs and soft/hard was tuned.
**It will not get there unless the capacity or the feature design itself (including per-candidate landing results in the features) changes.**

**Accounting for the tie band is not enough either (2026-08-19)**: since the teacher chooses randomly inside the band,
`match` measured by exact agreement drops regardless of the student's capacity. Remeasured by band agreement (band eps=0.1,
32 action bins, **with a random-shot control**), overall 10.1% → **33.6%**. But on ground where the control is 19.8%,
it is **only 1.7x**. And the ratio to control goes from 2.4x early → **1.4x late**:
**in the stretch where the board is decided, the student is nearly random**. The gate's scale was indeed too strict, but
it is not at the level of "able to imitate the teacher".
(Caveat: the checkpoint used is hidden=128, not the best of the sweep)

## Planned: RL (REINFORCE)

- Still too early. Plain REINFORCE easily breaks things when BC is shallow (confirmed in the past)
- Condition for adding it: `match` fairly high (roughly 60–70%+) and the student's `score` close to bootstrap
- How: only a short fine-tune after BC finishes (e.g. `--episodes 50 --lr 0.002`). Off by default
- Until then, thickening BC (collection size, epochs) comes first
