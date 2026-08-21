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

### Status: one game after the fall calibration (2026-08-20)

After calibrating `GRAVITY`, seed 642746 **dies at move 209** (269 moves before calibration).
The following are deterministic quantities of this one game, not score.

- **The cause of death is being stuck, not suicide.** 10 positions contain at least one lethal candidate, but
  **there are 0 moves where a lethal move was chosen while a surviving move existed**. At the last move, 209, all 37 candidates are lethal
- **Getting stuck comes in one move.** Surviving candidates go 51/56 at move 207 → 50/61 at 208 → **0/37** at 209.
  Before calibration they were whittled down over 10 moves; now there is slack right up to the end and then a sudden drop

### Won't do: deepen the lethal filter to two plies (2026-08-20)

The idea of discarding, before eval, surviving candidates where "wherever next is dropped, the game dies". On the game above,
**of 10 dangerous positions, 0 have candidates split by being stuck on the second move**.
At move 208, **all 50 surviving candidates are stuck on the second move**, so there is no room to choose at all.

The board drops from "50 surviving candidates" to "all 50 are dead ends" in one move, so
**the branching point is 3 or more moves earlier and in principle invisible to a two-move lookahead**.
The cost is running every x of next per candidate (over 30x the search), so it does not pay even if limited to dangerous positions.

*The physics before calibration gave the same conclusion (only 2 of 20 dangerous positions split,
and in both the chosen move was not on the stuck side).*

### Measuring fruits that never merge (fossils) (2026-08-20)

`scripts/fossils.py`. Gives each fruit an id and counts the moves until it merges away.
On seed 642746 (209 moves), **253 fruits are born, with median age 3 moves**. However,
**12 (4.7%) stay 50 moves or more**. Classified by the gap above them (in fruit-moves):

| age | touch | near | far | none | partner on board |
|---|---|---|---|---|---|
| 0-5 | 2.0% | 0.0% | 0.0% | **98.0%** | 58.2% |
| 5-20 | 13.6% | 3.9% | 1.8% | 80.7% | 58.1% |
| 20-50 | 11.9% | 20.1% | 6.6% | 61.3% | 58.8% |
| **50+** | **46.6%** | 16.9% | 2.7% | 33.8% | **66.1%** |

- **Short-lived fruits have nothing above them 98% of the time.** Having a roof clearly correlates with staying
- **Fruits that stay have a partner on the board in 66.1% of moves.** They are missed merges, not waiting merges.
  This is the supply-side hole in "merging cannot keep up with supply" (→[scattered low tiers late](#investigated-sudden-death-from-scattered-low-tier-fruits-late-in-the-game))
  a supply-side hole, **not yet addressed**
- **The numbers are diagnostics and must not be the basis for choosing a weight**
  (→[How to measure](#how-to-measure-traps-we-keep-stepping-in))

### Measured when adding a new term: perch (2026-08-20)

**Symptom**: on a board that has grown a pineapple + pear, cherries and strawberries are placed on top of that pile.
A small fruit once placed there stays 20+ moves without meeting a partner (the same fruit sits on the pineapple at moves 97-101
and on the peach at moves 103-122).

**The cause is a hole in the rules, not a weight.** No term looked at "a small fruit on top of a big fruit":

- `_bury_penalty` counts only the `over.type > under.type` side. The inverted
  shape of "a small fruit sitting on a big one" slips through
- `_size_order_penalty` excludes vertically stacked pairs with `abs(a.x - b.x) < min(r) * 0.5`
  as the same column
- The remaining sideways pairs are removed by `_size_order_exempt` too (back then **one partner anywhere
  on the board** granted the exemption, so the cherry at move 97 was exempted by its partner at the far edge, x=384.
  Now only a partner in the same valley counts)

Splitting move 97 (board `appl@51 pine@78 oran@145 grap@209 peac@222 pear@290 oran@324 grap@372 cher@384`,
held=cherry) by candidate, only the 4 that put it on the pineapple pile have eval 0.275 with every term near 0,
and all the rest carry bury 20.0 and score −18.8 or lower. **It is not a tie band; a single term decides the order**
(→[telling them apart in the tracing procedure](AGENTS.md#do-not-run-an-ab-while-obvious-blunders-remain)).
On a board where every placement buries something, only "putting it on top" was free.

**What was added**: `_perch_penalty` (→[rule list](#current-penalty-rules)).
The inverse of `_bury_penalty`; the check is not contact but "inside the big fruit's footprint, with its bottom
above that big fruit's center". The cherry at move 97 does not touch the pineapple; it sits in the valley between apple and orange
(gap to the pineapple top 46.3), so a contact-based check cannot pick it up.

**Weight sweep (150 positions / 5 seeds, deterministic per-position quantities)**:

| | master | w=4.0 | w=8.0 | **w=16.0 (adopted)** |
|---|---|---|---|---|
| Agreement with master | (baseline) | 86.7% | 81.3% | **80.0%** |
| Moves that created a perch | 38/150 | 30 | 25 | **21** |
| of which avoidable but remaining | 19 | 11 | 6 | **2** |

- **In 19 of the 38, every candidate creates a perch** (unavoidable), so that is the floor.
  w=16.0 fixes 17 of the 19 avoidable ones
- **From 8.0 to 16.0 agreement only drops from 81.3% to 80.0%, while fixes rise from 13 to 17.**
  Doubling the weight barely moves moves because the term acts as a binary filter
  with no sensitivity to its level (x0.5-x2.0 below)
- Cherry × pineapple becomes 64.0, above one bury (20.0) and close to foreign_aim (100.0).
  It is adopted knowing that this level **avoids perches even at the cost of rejecting a merge**
- The 20.0% change in moves is on par with fixing the bury window (21.6%)

**Does it escape the band** (254 positions, `--eps 0.1`, measured by adding `x0.0` to `band_escape.py`):

| Term | Moves change | Escapes the band |
|---|---|---|
| **perch x0.0** | **22.8%** | **22.8%** |
| foreign_aim x0.0 | 22.8% | 22.8% |
| bury x0.0 | 21.3% | 21.3% |
| size_order x0.0 | 18.1% | 16.5% |
| big_layout x0.0 | 19.3% | 3.5% |
| perch x0.5 / x1.5 / x2.0 | 4.3% / 2.4% / 3.1% | 3.9% / 2.4% / 3.1% |

**Every changed move escapes the band.** Like foreign_aim and bury it works as a binary filter of "applies or not",
and tuning the weight (x0.5-x2.0) moves only the usual 3-4%.
As [The existing weights have no leverage](#the-existing-weights-have-no-leverage) concluded, what worked
was not the weight but **the definition of what counts as a penalty**.

**A/B n=10 (seed 956314-, max_steps=400, A=no perch / B=with perch)**:

| Metric | A | B | Δ | t |
|---|---|---|---|---|
| score | 2173.10 | 2124.60 | −2.2% | −0.26 |
| steps | 223.2 | 215.5 | −3.4% | −0.49 |
| merges | 202.6 | 196.0 | −3.3% | −0.43 |
| cascades | 19.10 | 21.10 | **+10.5%** | 1.36 |
| max_type | 9.20 | 9.30 | +1.1% | 0.43 |

**It says nothing.** Not a single row is significant (the score 95% CI is [−464.8, +367.8],
paired SD=582.0, n=130 needed to speak to ±100 points), and the seed head-to-head is win 5 / loss 5.

- **The −2.2% is made by a single seed.** seed=956317 goes 3189 → 1780 (−1409);
  without it the mean is +102.7. The median difference is −3.5, essentially zero
- **Seeds reaching max_type 10 are 3 vs 3.** 956317 and 956321 drop from 10 → 9, while
  956318 and 956319 rise from 9 → 10. [When the vertical size order was reverted](#vertical-size-order-stage-gate-trapped-fruit-penalty-2026-08-18-reverted)
  the worsened seeds all went max_type 10 → 9, but that pattern does not appear here
- Only cascades is +10.5% (t=1.36), the one metric that leans plausibly

**How to read it**: n=10 only detects a large collapse, and none appeared. It is not evidence of improvement either.
The basis for keeping the rule is still the per-position quantities above and the fraction escaping the band.

## Open tasks

**Vision**
- `10.png`: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than policy / held

**Physics simulation**
- Friction between fruits seems low: fruits slide in far more than in the real game.

**Training pipeline**
- Training episode length: the default `max_steps=300` of `train_sim.py`, against post-calibration measurements
  (median 234 / p95 303 / max 320 moves, →[How to measure](#how-to-measure-traps-we-keep-stepping-in)),
  **truncates 5-10%**. **In training it is not merely a reporting bias: the REINFORCE
  return itself comes out missing**, so set it to 400 to match the other scripts.
  The cost barely rises (games over 300 are under 10% and the longest is 320, so
  removing the cap extends by at most 2 moves on average)

## How to measure (traps we keep stepping in)

**Score noise is very large.** This is the biggest wall for improving the policy, and every attempt below
got buried here. Read this section before reporting numbers.

- The per-game standard deviation is ~1000-1200, and even in paired comparisons on the same seed the SD of the difference is 78-496.
  Seeing ±100 points as significant needs **n≈100**. Tens of episodes are not enough.
  It is faster to look for a proxy metric with lower variance than score
- **At the default n=50 only changes of ±7% (±164 points) or more are visible** (the measured SD of the difference
  is 560-600). This is a screen accepted knowingly, and changes that do not reach it are not added
  ([AGENTS.md](AGENTS.md#when-touching-the-policy-or-training)). Many 1-3% terms have been
  measured so far, and not one became significant. **Rather than raising n to catch them,
  it is faster to look for terms that move moves out of the band** (→[screen](#screen-on-does-it-escape-the-band))
- **Pairing barely helps (2026-08-21).** The score correlation between A and B on the same seed
  is **r=+0.11 / +0.18** (two measurements at n=300), a gain in SE of only 6-10%.
  Changing one move makes the board diverge, so even with matched seeds they are nearly uncorrelated.
  The point of pairing is not variance reduction but **matching the draw luck on both sides**.
  So **side A may be reused across variants** (`scripts/compare_b_only.py`)
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
  the better the change the more it is underestimated. **Natural game ends retaken after the fall calibration, n=60:
  mean 239 / median 234 / p90 290 / p95 303 / max 320 moves** (`eval_policy.py
  --max-steps 400` gives `truncated=0/60`). `--max-steps` truncates **0% at 320,
  5-10% at 300** (p95=303). The default 400 stays. At 200 it is 64%,
  **at 100, 100% are truncated** (the 08-03 accident was this). `compare_policy.py`
  warns if even one is truncated
  - Setting the cap needs the tail, not the mean, so `eval_policy.py` also prints quantiles.
    **With one run that has zero truncations, the truncation rate for any cap can be read from it**
    (do not rerun per cap)
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
nothing.** The tie-break of the time (the third decimal of bumpiness) was doing no work and was later
[retired](#retired-bumpiness-height-variance-2026-08-21). What decides the order inside the band now is
`center_tiebreak`, but **given this result, do not look for meaning in that ordering.**

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
- **Inside the band every large term is saturated.** Inside the band (eps=0.1) the only term that differs between candidates is
  the ideal part of `size_order`; bury / perch / foreign_aim / excess_same /
  the pair part of size_order are completely flat inside the band in 99.7-100% of positions
  (they do the job of knocking candidates out of the band)
  (→[Split composite terms into sub-terms](#split-composite-terms-into-sub-terms-2026-08-21). The original numbers in this section
  date from when `packed` existed and `perch` did not; they were retaken with the current terms)
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

**Whether a new term is worthwhile can be read from the same table.** With the term added, looking at `x0.0` (cutting that term)
puts it on the same footing as the existing weights. The measurement when adding `_perch_penalty`
is an example ([measured](#measured-when-adding-a-new-term-perch-2026-08-20)).

#### Do not penalize board properties the current move cannot change (2026-08-21)

**Terms that do not escape the band share a shape: they measure "a board property the move cannot change".**
eval scores the post-drop board per candidate and compares them, so a term with the same value for every candidate
**just adds equally to every candidate and does not change the ranking one bit**. No matter how much you multiply the weight.

Shapes the dropped fruit itself creates (a new inversion in `so_pair`, the lid in `bury`, the shoulder in `perch`,
the fruit directly below in `foreign_aim`) change per move, so they can decide the ranking. How scattered the whole board is
does not change. **This one point explains the retired terms**:

| Term | What it measured | Does one cherry move it | Band escape |
|---|---|---|---|
| bumpiness (`_height_variance`) | variance of heights across the board | barely | capped at 7.0%, A/B null |
| big fruits not close enough (`bl_cluster`) | spacing between big fruits | no | ─, A/B null |
| same-type scatter (`same_type_pull`) | left-right scatter of same types | only when it is the same type | 0.7%, A/B −2.9% |

**Do not put `so_ideal` (deviation from the assigned seat) in this group.** With 0.0% band escape it looks the same,
but it drops for a different reason (→[Terms divided by an average](#terms-divided-by-an-average-thin-out-as-the-board-fills-2026-08-21)).
**It was nearly retired and then restored.**

**The idea of penalizing "same-type big fruits scattered left and right" fails for the same reason** (considered on 2026-08-21).
Two peaches split left and right stay split wherever the cherry is dropped, so
every candidate gets the same value. As a description of the board it is right, but it is no material for choosing a move.

On the same position set (275 positions, same types of tier ≥5 more than 150px apart across the center), splitting the difference between the best move
on the left half and the best move on the right half per term, what decides it is `perch` (contribution \|mean\| 32.9) and
`so_pair` (21.7), both looking at **the shape the fruit placed now creates itself**.
`corner_pocket` has zero left-right difference in 98.5% of positions (it fires only when the biggest fruit is wall-anchored).
**29.5% of positions are ties within 0.1 points between left and right**, and which side to grow is not in the current eval.

#### Terms divided by an average thin out as the board fills (2026-08-21)

**`so_ideal` was nearly retired and then restored. The screening was not looking at the early game.**

The ideal deviation in `_size_order_penalty` is `sum(|x - ideal_x|) / fruit count * 0.004`.
The denominator is the fruit count, so **the emptier the board, the stronger it is**:

| Fruits on the board | ideal term | relative to the 0.1 tie band |
|---|---|---|
| 1 | 1.367 | **13.7x** |
| 3 | 1.154 | 11.5x |
| 10 | 0.631 | 6.3x |
| 20 | 0.564 | 5.6x |

Both `band_escape.py` and the screening in `scripts/` default to **`--skip 60` (move 60 onward)**, and
**collect only the fully thinned-out late game**. There band escape came out 0.0%, but the early game was a different story.

Removing it breaks 4 tests in `test_policy.py`, and **in all 4 the chosen move becomes x=204 (the center of the board)**.
On a sparse board the only term left is `center_tiebreak`, and everything drops in the center.
`test_leaves_room_for_missing_rung_between_neighbours` had its order go from `[2,3,4]` to
**`[4,3,2]`, reversed**. `so_pair` needs at least two fruits to work, so the early-game size order
is made by this term alone.

**The early metrics of the A/B were saying so** (`so_ideal_off`, n=300):

```
score        2303.51 -> 2307.81  (+0.2%)  t= 0.13     ← final score does not move
early_score   237.05 ->  230.99  (-2.6%)  t=-4.12 *
early_crown    346.2 ->   340.9  (-1.6%)  t=-4.21 *
```

**Reading these two as "they do not correlate with score, so they are not proxies" and moving on was the mistake.**
Being unusable as a proxy is separate from the fact that early-game behavior changed.
**Read a significant early metric in itself as evidence that "something changed".**

The weight can stay as it is. Sweeping 0/5/10/30x of 0.004 at n=50 gave
`+0.2 / +1.2 / +8.1 / −1.5 %`. Only x10 jumping is noise, not an effect
(a real effect would grow along with the weight). A version applied only to the biggest fruit and one tier below is also null at −2.8%.

### The existing weights have no leverage

What decides which candidates enter the band is the large terms, so their weights were swept.
Below was retaken **after the fall calibration** (459 positions, 6 seeds, `--eps 0.1`, median band size 6).
x0.0 is with that term cut.

| Term | x0.0 moves change | x0.0 outside the band | x0.5 | x1.5 | x2.0 |
|---|---|---|---|---|---|
| perch | 27.7% | **27.7%** | 4.8% | 4.1% | 6.1% |
| bury | 17.2% | **17.2%** | 4.4% | 3.3% | 5.7% |
| foreign_aim | 16.1% | **16.1%** | 0.2% | 0.0% | 0.0% |
| size_order | 15.9% | 14.8% | 4.1% | 2.4% | 3.9% |
| variance | 58.2% | 12.9% | 0.4% | 0.4% | 1.1% |
| big_layout | 18.5% | 5.7% | 0.7% | 0.4% | 0.9% |
| excess_same | 3.3% | 3.3% | 1.1% | 0.2% | 0.9% |

- **There is still no leverage on the weight-tuning side** (at most 6.1%). Even
  `drop_ideal`, which changed 50% of moves, was null at n=133, so measuring this range finds nothing
- **The 3 terms that work (perch, bury, foreign_aim) all have "moves change = escapes the band".**
  That they act as binary filters of applies-or-not also shows from the cutting side.
  `foreign_aim` stays at 0.2% or less at both 0.5x and 2x; the weight 100.0 is so large that
  candidates it applies to are out of contention from the start
- **`variance` moves 58.2% for 12.9%, `big_layout` moves 18.5% for 5.7%.**
  Continuous quantities only swap candidates inside the band (`variance` was
  [retired](#retired-bumpiness-height-variance-2026-08-21) with this table as one reason. The table is kept as measured at the time)
- **`excess_same` gives only 3.3% even when cut.** Despite being a heavy penalty of 20.0 for 3+ of the same type,
  it barely decides the chosen move. It likely saturates by taking the same value for most candidates
  (the same shape as "discrete quantities saturate" in [What the band actually looks like](#what-the-band-actually-looks-like))
- **This table keeps its order and structure across the calibration that halved gravity (GRAVITY 2800 → 1400).**
  Before calibration it was perch 28.3 / foreign_aim 21.3 / bury 20.6 / big_layout 2.1 /
  excess_same 1.5%, so **whether there is leverage is decided by the shape of the policy, not by the physics**

**Conclusion: bootstrap is somewhere weight tuning cannot get out of.**
The inside of the band is indifferent, the weights deciding who enters the band have no leverage, new count terms do not move
inside the band, new continuous terms only move inside it, and the road of a deeper search was
[measured and shelved on 08-17](#run-cost-faster-physics-and-search-width-2026-08-17).
To go further it is either **the definition on the side that decides who enters the band** (not weights, but what counts as a penalty), or
an evaluator that sees differences the current features cannot (a learned value function).
This matches how [Policy (bootstrap) design](#policy-bootstrap-design) has positioned it from the start:
"a thin policy before RL".

### Split composite terms into sub-terms (2026-08-21)

Of the 7 terms in the table above, `size_order` and `big_layout` are **sums of two rules of different nature**,
and measured coarsely one part's work is buried in the other's noise. They were split and remeasured
(459 positions, 6 seeds, median 43 candidates. The position set is built the same way as in `band_escape.py`, and
the sum of sub-terms was checked to match the real eval on every position).

**A term that does not spread between candidates cannot choose a move however its weight is tuned** (it vanishes in argmax).

| Term | candidate range (median) | all candidates equal | nonzero candidates | \|value\| mean |
|---|---|---|---|---|
| foreign_aim | 100.00 | 0.4% | 18.3% | 18.30 |
| perch | 48.00 | 9.6% | 80.8% | 94.01 |
| size_order pair | 48.00 | 1.1% | 73.8% | 34.93 |
| bury | 20.00 | 28.8% | 88.9% | 53.86 |
| size_order ideal | 0.36 | 0.9% | 75.4% | 0.30 |
| excess_same | **0.00** | **60.6%** | 69.2% | 41.10 |
| big_layout corner pocket | 0.00 | 77.8% | 19.8% | 24.09 |
| big_layout not close enough | **0.00** | **81.0%** | 16.0% | **0.14** |

- **`excess_same` "applies rarely but hits hard when it does".** Its median candidate range is 0.00, and
  in 60% of positions all 43 candidates are equal, but **in the remaining 39.4% it moves between candidates**. When it moves,
  its weight is a heavy 20.0 per excess fruit. Looking only at the median and reading "a constant offset, so dead"
  is wrong (it was read that way once and refuted by an A/B; see the [measurement](#measured-excess_same-was-kept-2026-08-21) below).
  The 3.3% band escape in the table above means "rare", not "powerless"
- **The not-close-enough part of `big_layout` has a |value| mean of 0.14 points.** Only 16% of candidates are nonzero, and
  "exempt for the diameter of the missing type" eats up almost all of this term's output. Its other half,
  the corner pocket, is rare with 19.8% nonzero but is a binary filter averaging 24 points. The coarse table's
  "moves 18.5% for 5.7%" was **these two seen separated**.
  Not-close-enough was later [put through an A/B and retired](#retired-big-fruits-not-close-enough-2026-08-21)
- **Every term that works is, without exception, a binary applies-or-not.** Continuous quantities only swap
  candidates inside the band and cannot push them out

**Restricted to inside the band (eps=0.1), no board penalty can make a difference.**

| Term | range inside the band (median) | all candidates equal inside the band |
|---|---|---|
| size_order ideal | 0.00 | 55.6% |
| big_layout corner pocket / not close enough | 0.00 | 83.5% / 85.0% |
| everything else | 0.00 | 99.7-100% |

**A term that can make a difference inside the band becomes a de facto tie-breaker, whatever you meant to write there.**
At the time of measurement that role was played by bumpiness (range inside the band 0.01, nonzero in every position), and on that
basis it was [retired](#retired-bumpiness-height-variance-2026-08-21). Now `center_tiebreak`
by definition always takes a different value per candidate, so it carries the role explicitly.
**When adding a new continuous term, check that it has not fallen into this position.**

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

### Measured: excess_same was kept (2026-08-21)

A/B cutting `_excess_same_penalty`, n=100, 0 truncated. **Cutting it leaned toward worse.**

| Metric | A (with) | B (cut) | Δ | t | 95% CI |
|---|---|---|---|---|---|
| score | 2329.90 | 2281.26 | −2.1% | −0.84 | [−164.0, +66.7] |
| merges | 214.4 | 209.7 | −2.2% | −0.97 | |
| cascades | 21.83 | 21.41 | −1.9% | −0.71 | |

win/loss 43/55/tie 2, paired difference −48.6 (SD of the difference=581.2). **Not significant** (CI crosses 0), but
unlike bumpiness (+1.5%, a 52/48 coin toss) every sign is negative. → **Keep it**.

**Lesson**: in [the sub-term table](#split-composite-terms-into-sub-terms-2026-08-21) the median candidate range was 0.00, so
it was read as "constant offset = dead", but **the median makes you misread "rare" as "powerless"**.
It moves in 39.4% of positions, and when it moves the weight is 20.0. The same shape, the corner pocket of `big_layout`
(19.8% nonzero, mean 24 points), had been rated "working", so the judgments were inconsistent.
**Look at the nonzero rate together with the size when it moves.**

Redrawn at n=300: **−2.2% (t=−1.60, win/loss 139/155)**. Not significant, but
it matches n=100's −2.1% in sign and size, with merges −2.0 / cascades −2.2 / steps −1.7 /
max_type −0.6%, every metric in the same direction. **The decision to keep it stands.**

### Measured: one night of A/B runs (2026-08-21)

Every existing term was cut and measured in turn. **Not one deserved retirement.**

| Variant | What it did | n | score | t | Verdict |
|---|---|---|---|---|---|
| `so_pair_off` | cut left-right size inversions | 50 | **−7.0%** | −1.61 | **keep** (cascades −11.3%, early_crown −4.7%, both CIs do not cross 0) |
| `excess_same_off` | cut 3+ of the same type | 300 | −2.2% | −1.60 | **keep** |
| `valley_grow_off` | cut the valley-growing bonus | 300 | −1.7% | −1.12 | **keep** (cascades −2.8%) |
| `center_off` | cut the center tie-break | 50 | −2.6% | −0.66 | **keep** |
| `so_ideal_off` | cut the assigned-seat deviation | 300 | +0.2% | 0.13 | **keep** (→[Terms divided by an average](#terms-divided-by-an-average-thin-out-as-the-board-fills-2026-08-21)) |
| `same_type_pull_x5` | **new**. penalize same-type scatter at 0.1/px | 50 | −2.9% | −0.75 | **not added** |

**The band escape prediction held.** In screening on 450 positions only
`so_pair` 7.3% and `center` 8.9% passed the threshold, and the only one that actually moved a lot was `so_pair` (−7.0%).
The rest were 0.0-1.8% and every A/B was null. **Looking at band escape before the A/B tells you which will move.**

**But band escape only looks at the late game** (`--skip 60`). Terms that work early do not show up here
(`so_ideal` is one).

The methodology measured the same night is in
[Pairing barely helps](#how-to-measure-traps-we-keep-stepping-in) (reusing side A).

### Retired: big fruits not close enough (2026-08-21)

A/B cutting the second half of `_big_layout_penalty` (a continuous quantity penalizing gaps between big fruits),
n=100, 0 truncated. score 2335.57 → 2404.60 (+3.0%, t=1.10, CI [−55.5, +193.5]),
merges +2.9%, cascades +2.0%, win/loss 52/48. **Not significant, but the signs line up on
the "better cut" side**, consistent with its mechanism of a |value| mean of 0.14 points → deleted.

It was a term eaten by its own exemption clause. Big fruits usually line up in size order, so between adjacent pairs
there is exactly a "missing type", and exempting that diameter makes the gap vanish.
**Nonzero for only 16% of candidates, and 0.14 points when it moves** (→[sub-terms](#split-composite-terms-into-sub-terms-2026-08-21)).

The remaining corner pocket part was renamed `_corner_pocket_penalty`. **Putting two rules of different nature under one name
buries one part's work in the other's noise and gets misread** — both this section and
[excess_same](#measured-excess_same-was-kept-2026-08-21) are examples.
`BIG_CLUSTER_SPAN` was only used by not-close-enough, so it went too.

**Note**: measurements of `big_layout` appearing earlier in NOTES (5.7% outside the band and so on) are
**from when the 2 rules were combined**. They are kept as the record of that time.

### Retired: bumpiness (height variance)(2026-08-21)

An A/B cutting `_height_variance` (spread of crowns per column bin) at n=100, 0 truncated, is
null. score 2312.42 → 2346.60, t=0.53, CI [−94.5, +162.8], win/loss 52/48, and
the sign if anything favors cutting it. **The idea of "penalizing bumpiness" was measured and dropped.**
The reason is that it is continuous — it only swaps candidates inside the band and cannot push them out
(→[Split composite terms into sub-terms](#split-composite-terms-into-sub-terms-2026-08-21)).
`DANGER_Y` and `_top_crown` were only used for relaxing it, so they went too. Tie ranking is
carried explicitly by `center_tiebreak`. Details in `git log -- src/penalties.py`.

### Replaced the dangerous height slope with a filter (2026-08-20)

**Trigger**: "isn't the dangerous-height rule unnecessary? It is always close to the edge right before a double watermelon".

**How close to the edge**: the trigger line `DANGER_Y` 70.9 against the losing line `GAME_OVER_Y` 14.9.
The 56.0 difference is 11.2% of the board height 500 (it only starts at 85.8% stacked from the floor),
a width that cannot fit one apple (diameter 98.6). The penalty is at most 28.0 even right at the death line,
smaller than a watermelon merge 55 / double watermelon clear 65. **It never stops a climb by rejecting a merge.**

**Measured per candidate** (428 positions, 19287 candidates, move 60 onward on 6 seeds × 240 moves):

| | rate |
|---|---|
| candidates with nonzero `danger` | 21.4% |
| lethal candidates (crown < 14.9 after the drop) | 1.1% |
| positions with at least 1 lethal candidate | 30 (7.0%) |
| chosen move has nonzero `danger` | 19.9% |
| **chosen move is lethal** | **5 (1.2%)** |

**It committed suicide even with the slope in place.** In those 5, a lethal move won even though 30-45 surviving candidates existed,
with differences of 8.5 / 24.3 / 40.9 / 121.5 / **261.3**. Not merge points but
`bury`, `excess_same` and `size_order` made the difference (the lethal side's merge points are 0-10).
**A slope capped at 28 cannot reach that, and no amount of finite penalties can exceed 261.**

Meanwhile, cutting the slope made 5 cases where a lethal and a surviving move **tie at 0.00 difference** fall to the lethal side
(10 in total, 2.3%). In other words the slope only worked "as a tie-break".

**What was added**: a lethal-move filter in `choose_x`. If there is even one surviving candidate,
lethal candidates are dropped before comparing eval. All 10 cases are fixed. On a board where every candidate is lethal
(stuck) it picks the best move as before. These two behaviors are pinned in `tests/test_policy.py`
(the self-destructing board is move 172 of seed=982108, the stuck one move 214 of seed=221700.
Both are real positions picked from full playthroughs).

**What was removed**: the slope `(DANGER_Y − crown) × DANGER_CROWN_WEIGHT` and
`DANGER_CROWN_WEIGHT`. `DANGER_Y` remains as the threshold that relaxes bumpiness.

**Effect in full playthroughs** (6 seeds × 240 moves, compared with master on the same seeds):

| | master (slope only) | filter |
|---|---|---|
| moves | 1211 | 1299 |
| chose a lethal move while a surviving move existed | **6 (0.5%)** | 0 (the filter blocked 35 moves, 2.7%) |
| stuck positions (every candidate lethal) | 0 | 4 (0.3%) |
| deaths | **6 / 6 games** | 4 / 6 games (2 survived to the 240-move cap) |

**All 6 deaths on master were avoidable.** Zero stuck positions = even at the moment of death
a surviving move remained, yet a lethal one was chosen. All 4 deaths on the filter side are stuck positions,
with zero suicides. The filter firing 2.7% more than suicide at 0.5% is because removing the slope
made lethal moves rise to the top more easily (the filter takes over what the slope was holding back).

steps 1211 → 1299 and score 11961 → 12862 also came out, but **these are n=6 full playthroughs, so they are not evidence**
(→[How to measure](#how-to-measure-traps-we-keep-stepping-in)). Only the structural metrics above are read.

**Score was not measured.** The band escape of the slope alone is 4.7% (eps=0.1) /
3.0% (eps=0.5), in the null zone of [the screen](#screen-on-does-it-escape-the-band)
(`drop_ideal` 6.3% → null, bumpiness x4 7.0% → null). **No A/B was run**.
The only basis is the per-position quantities above.

**Note**: no term looks at absolute height any more. The only deterrent to height is "does the next move die",
which works only two plies ahead. Whether a pathology of walking into a stuck board by itself has appeared should be checked
in [the fixed-point observation](#current-approach-fixed-point-observation-of-seed-642746-2026-08-19).

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

### Deciding perch acceptance by "orange or bigger" (2026-08-20, rejected)

The idea of changing the criterion `_perch_penalty` uses to decide which fruits may sit on top from **the type gap to the biggest fruit** (current,
`PERCH_MIN_GAP` 5) to **the fruit's own type** (orange or bigger is fine).

**Trigger**: move 34 of seed=956317. A dekopon (3) sits on the left shoulder of a peach (7),
and at move 40 that dekopon ate the orange on the bottom rung of the ladder and crushed it. The type gap is 4, so the current rule
does not fire (perch is 0.000 in that position, and the shoulder side wins with size_order 6.078 against bury 7.0).
With an absolute threshold, move 34 sends the dekopon to the right (368,431), and the left pile stays as the ladder
`peac@69 / oran@160 / pear@226`. Agreement over 150 positions was also nearly the same,
80.0% → 79.3%.

**A/B n=10 (seed 956314-, A=current / B=proposal)**:

| Metric | A | B | Δ | t |
|---|---|---|---|---|
| score | 2124.60 | 1902.90 | −10.4% | −1.14 |
| steps | 215.5 | 202.2 | −6.2% | −0.85 |
| merges | 196.0 | 179.9 | −8.2% | −1.01 |
| cascades | 21.10 | 18.60 | −11.8% | −1.08 |
| **max_type** | **9.30** | **8.70** | **−6.5%** | **−2.71** ← 95% CI [−1.1, −0.1] |

**Every metric is negative, and only max_type is significantly negative.** With n@5% of 9, max_type is one of the few
metrics n=10 can speak to, and it dropped. Worsened seeds: 956319 went 10 → 8, 956314 10 → 9,
956320 9 → 8, 956318 10 → 9.
The same shape as [when the vertical size order was reverted](#vertical-size-order-stage-gate-trapped-fruit-penalty-2026-08-18-reverted)
(every metric negative, max_type of the worsened seeds drops), with symptoms stronger than then.

**Assessment**: the proposal **locks dekopons out of every big fruit's shoulder**, but a dekopon on a peach's shoulder
is also the correct place for building the next rung. Move 34 was a blunder not because it sat on the shoulder but
**because at move 40 that dekopon ate the orange on the bottom rung of the ladder**; the shoulder was not the cause.
Forbidding it sends mid-size fruits to the other side and the ladder does not grow. On top of that the proposal also loosens toward
**missing an orange on a melon's shoulder** (it allows type gap 5), so it lost on both the tightening and the loosening side.

**Remaining task**: the symptom of move 34 itself is unresolved. To pursue it,
the term should look not at "sitting on a shoulder" but at "a merge that eats the bottom rung of a ladder".

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
- **Widening bury's vertical window to the partner's diameter** (`0.6r` → `2r`): **6.1%** escape the band,
  in the null zone. Per position, agreement is 89.9%, and in the 24 changed cases the intended misses halve but
  **ordinary burying increases** (the band escape of `bury` itself drops from 20.6% → 14.7% = it just took over
  the work). **And after calibration the premise disappeared**: the fraction of fossils under a touching roof went
  from 21% → 46.6%, so the current window already sees them. *Measured before calibration*
- **Measuring big-fruit proximity by center distance instead of x gap**: the cluster term reads vertically stacked pairs as
  "gap 0" (17% of same-type big-fruit pairs are this shape). Fixing it gives 5x the value late in the game, but **it changes 16.0% of moves
  and escapes the band 1.4%**, and 0.0% when the weight is tuned. The same shape as `big_layout` itself:
  **as long as it is continuous it only swaps inside the band**. To make it work, it has to be not a distance but
  a binary condition "does this move split a big-fruit pair". *Measured before calibration*

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
| valley-growing bonus | `valley_grow_ok` | landing in a valley whose fruit is the same type as held / whose fruit is one above held with held and next the same type | **−3.0** (bonus) |
| merge pushed to the big side | `merge_lands_big_side` | moves where the fruit made by the merge (held's lineage) stops **at least one radius of the dropped fruit** toward the big side of the drop column | **−0.5** (bonus) |
| stranded | `stranded_drop_penalty` | the dropped fruit (held's lineage) stops in **a valley of fruits 2 or more types bigger**, **with no partner in the same valley**. Applies to merging moves too | `STRANDED_DROP_WEIGHT` 20.0 × (type gap − 2 + 1) |
| center tie-break | `center_tiebreak` | distance between the drop column and the center. **A term only for ordering**, it does not express how good a move is | `CENTER_TIEBREAK_WEIGHT` 0.001 (max 0.19 < minimum merge score 1.0) |

**These two bonuses are mutually exclusive**. Valley growing applies only **when held itself did not merge**, and the big-side merge
only **when it merged** (`held_merged`, not the merge count `merges`, so that an unrelated merge elsewhere on the board
does not grant the exemption). The big-side weight of 0.5 is set as the cap that does not overturn the smallest
merge score difference of 1.0 (cherry→straw). The property that the ranking between merges is decided by
the real game's score is not broken.

**Board-wide penalties (`board_penalties`, on the post-drop board every time)**

| Rule | Function | Content | Weight |
|---|---|---|---|
| burying | `_bury_penalty` | how much merge-candidate fruits are covered by other types. The contact window is based on **both radii** `(under.radius + over.radius) × 0.9` (based on the lower fruit alone, the window narrows the more a big fruit sits on a small one and it escapes detection) | fruit with a partner on the board `BURY_WEIGHT` 20.0 / without `BURY_LONE_WEIGHT` 15.0 |
| perch | `_perch_penalty` | small fruits inside the footprint of a big fruit (from the biggest down to `PERCH_BIG_SPAN` 1 tier below) with their bottom above that big fruit's center. Counts the amount by which the type gap exceeds `PERCH_MIN_GAP` 5 (up to orange on a pineapple's shoulder is 0, dekopon 1 / grape 2 / strawberry 3 / cherry 4). Contact is not required, so shapes sitting on the pile with one tier in between are caught too | `PERCH_WEIGHT` 16.0x |
| excess same type | `_excess_same_penalty` | 3 or more of the same type (up to 2 are allowed as waiting to merge) | 20.0 per excess fruit |
| size-order inversion | `_size_order_penalty` | pairs whose size order is inverted left to right (only fruits stuck in a valley of bigger fruits **and with a same-type partner left in the same valley** are exempt = `_size_order_exempt`. A partner outside the valley is blocked by the big wall fruits, so it does not exempt). **Exempt on moves where held merged** (that hole is closed by `merge_lands_big_side` and `stranded_drop_penalty` above) | pair difference×1.5 + ideal_x deviation×0.004 |
| corner pocket | `_corner_pocket_penalty` | the biggest fruit is on the big-side wall, yet there is a small fruit outside and below it (a fruit that gets behind L cannot meet its partner) | 50.0×(1+0.05×type gap)+depth×0.15 |

**Not a penalty: the lethal-move filter (`choose_x`)**

Candidates whose post-drop board crosses the losing line (`GAME_OVER_Y` 14.9) are discarded before comparing eval
as long as there is even one surviving candidate. It is not a penalty, so it is not in the table above
(→[Replaced the dangerous height slope with a filter](#replaced-the-dangerous-height-slope-with-a-filter-2026-08-20)).

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
