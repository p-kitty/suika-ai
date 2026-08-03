# Known issues and things to fix later

## Deferred

- **`10.png` vision**: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than policy / held
- **Training episode length**: raise `max_steps` and lower `episodes` (fewer, longer games)
- **Packing too tight early**: neighbors by size order and proximity (e.g. apple and dekopon) have too little gap between them. When stacking an orange only order-breaking moves remain. Some space is wanted
- **Big draws after the floor fills**: after the floor fills, big draws such as orange / dekopon are placed on the small side instead of the L (biggest fruit) side and the board collapses. The big/small side placement stops working partway through

## When to move

- Do not decide x on a moving board. Waiting for it to settle takes priority over lookahead (`src/settle.py`)
- Wait for creep not only on instantaneous velocity but also on sideways drift while quiet

## Policy (bootstrap)

- `src/policy.py` is a thin policy before RL. Only merging, dangerous height, burying, light size order and accident prevention for rolling / knock-aways
- The physics of falling, collision and merging is pymunk (`src/sim_physics.py`; UT in `tests/test_sim_physics.py`). `choose_x` scores with the same `simulate_drop`
- Moves are scored as `eval = score - penalties`. The only bonus is the real game's score; dangerous height, accidents and burying are penalties
- next lookahead: only the top `HELD_TOP` by held eval are re-evaluated with candidates at spacing `NEXT_CANDIDATE_STEP` (multiplied by `NEXT_DISCOUNT`). The physics is heavy, so it is coarser than held
- Burying is the main penalty. Moves that block a same-type pair waiting to merge with a bigger fruit of another type, directly above or on the shoulder, are heavily penalized
- Aiming at the center of a different type (`FOREIGN_AIM`): `FOREIGN_AIM_PENALTY` if the fruit directly below is a different type and in its center band (`FOREIGN_AIM_CENTER_FRAC`). OK if it is the same type. Not cut by `merges` (closes the loophole of rolling off a different type and merging). Excess same type (`EXCESS_SAME` = 20 per excess fruit). Stacking a different type in valleys or on shoulders is not forbidden
- Valley growing for big fruits is limited to when the valley has a same type, or held/next are both one smaller than the walls. Other gap filling gets the usual penalties (`GAP_JUNK` stays retired)
- Layout: big fruits stay close together. On the big side (`sign`), the corner pocket outside an edge-anchored L and below L's center is heavily penalized (`_big_layout_penalty`). `wrong_side_roll` is also for rolling accidents onto the same big-side floor
- Not included: push-in merges, restoring pushes, cascade gap opening, forced moves one tier up, hard-coded ladder firing
- Do not add UTs for concrete procedures. When something breaks, look at accident prevention or the observation side

## Ladder (firing a corner big fruit up a staircase)

A pear next to the inside of a corner peach, an apple and an orange on the **shoulders** of those two, firing with the final orange
to cascade 4→5→6→7. The same holds from a corner pineapple or corner melon onward; the staircase always goes down to the biggest drawable
(orange). But **it is not a shape to aim for every time**. It is an option when L is fairly big and the floor
is filled; from peach to pineapple it is often grown normally from the side.

For now it is **detection only** in `_ladder_anchor` / `_ladder_rungs`, not connected to move selection.
`SUIKA_LADDER=0` cuts detection entirely, but ON/OFF does not change moves.

What measurement has shown (`scripts/compare_policy.py`, a probe of 6 seeds × 120 moves):

- **Firing needs no guidance**. Once a ladder is built, `choose_x` ties with the best of an exhaustive
  sweep over x. The same-type contact points of `_add_near_fruit_x` are enough. Adding candidate x for firing
  was a no-op with every seed tied (measured and removed)
- **`FOREIGN_AIM` is unrelated to the ladder**. The rungs sit on shoulders, so this penalty never applies in the first place.
  "The -100 for directly above a different type crushes ladders" is wrong. It is physically never stable directly on top
- **`_size_order_penalty` is not in the way either**. Measured on ladder boards it is only 0.14-0.49
- **Without a filled floor the shape does not hold**. The pear is pushed out like a wedge and self-destructs, and wherever you drop
  you get only one rung (15 points). Filling the floor to the right edge gives 100 points. A filled floor is a gate condition
- **The bottleneck is building it**. A board with all 4 rungs appears only 12 times in 720. It is not that it cannot fire,
  but that it never gets built. This is where to intervene (as a board potential on the
  `_board_penalties` side, gated by a packed floor)
- Treating a rung as "a move that drops and places it" is a poor approach. Draws go up to orange, and
  the pear and apple rungs can only be grown by merging. Written as a placement condition it passed only 4 times
  in 400 boards
- The search cost is essentially the number of `simulate_drop` calls. `HELD_TOP` / `NEXT_CANDIDATE_STEP` decide the run time (the old 8/16 took 3.8 seconds per move and collection could not keep up. 2/32 gave 1.2 seconds and score -3.4%)
- Do not make `CANDIDATE_STEP` coarser. At 20 the spot directly above a dangerous pile lands on the grid and `test_avoids_dangerous_tall_stack` fails. Speed is earned on the lookahead side
- Cutting `SLEEP_FRAMES` does not work. A single `choose_x` gets faster, but the board settles differently and later moves get heavier, so the whole episode is actually slower (measured at 25). The physics fidelity (shared with `SimEnv`) also drops

## Training

- **score / penalties**: `score` is the real game's merge score (1-65, no penalties), `penalties` are the penalties for accidents and bad moves. `eval = score - penalties` is **only for bootstrap move selection** (`choose_x`). The student's quality, saving and logs use the real game's `score` (moves on ties). **The RL reward is score too** (dense penalties are not rewards)
- `src/reward.py`: `merge_score(merge_types)` gives only merge points identical to the real game (cherry→0 … watermelon 55, double clear 65). No survival bonus or death penalty. Episodes end as before (losing line / double clear)
- `src/encode.py`: fixed-length observation vector
- `src/sim_env.py`: headless drop sim (`sim_physics.simulate_drop`). `SimStep` is the real game's `score` only (no cumulative eval)
- Evaluation: `python scripts/eval_policy.py` (`--policy bootstrap|learned`. `--workers` default = logical cores/2)
- A/B: `python scripts/compare_policy.py`. Runs two bootstrap variants **on the same seeds**,
  reporting not just means but per-seed wins and losses and per-phase metrics (`early_score` / `early_crown` /
  `dead_early` / `cascades`). If every seed ties it warns "the change is not firing".
  Put changes through this before touching the policy, so nothing gets discarded on "it somehow got weaker"
- Training: `python scripts/train_sim.py` (collect → offline BC. The default max-steps=100 is a cap, not the losing line). best is score → moves → match
- Teacher collection runs in parallel with `ProcessPool` (default workers=logical cores/2; 8 on a 9700X; `--workers 1` for serial)
- `src/agent.py`: MLP with 32 discrete column bins / hidden 128 (old 20/64 npz files need retraining)
- Live play: `python main.py` (defaults to learned if an npz exists. `L` toggles bootstrap, `--policy bootstrap`)

## Planned: RL (REINFORCE)

- Still too early. Plain REINFORCE easily breaks things when BC is shallow (confirmed in the past)
- Condition for adding it: `match` fairly high (roughly 60–70%+) and the student's `score` close to bootstrap
- How: only a short fine-tune after BC finishes (e.g. `--episodes 50 --lr 0.002`). Off by default
- Until then, thickening BC (collection size, epochs) comes first
