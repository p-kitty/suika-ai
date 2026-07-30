# Known issues and things to fix later

## HELD / NEXT disappear and play stalls (near the top edge of the board)

- **Symptom**: when a pineapple sits where it almost sticks out the top, HELD and NEXT stop being detected and auto play stalls
- **How to reproduce**: positions with a big fruit (pineapple and so on) near the top edge of the board
- **Where to look**: `src/vision/held.py` (fruits sticking out at the bottom of the band)
- **Noted on**: 2026-07-30
- **Status**: fixed — blobs touching / near the bottom of the band are removed from the band mask (`_without_overhang`). Pinned in `tests/test_held.py`

## Placement does not account for NEXT / HELD

- **Symptom**: e.g. in a position wanting to grow a grape, with NEXT and HELD both strawberries, it does not put them on the grape
- **Suspected cause**: the policy / evaluation centers on "the fruit held now" and does not sufficiently consider cascades and growing based on NEXT and HELD
- **Where to look**: `src/policy.py`, placement scoring / candidate evaluation
- **Noted on**: 2026-07-30
- **Status**: fixed — when held/next are the same type, prefer the side where the growing target lines up / pushing to the big side (not directly above the center of a different type). Pinned by position UTs in `tests/test_policy.py`

## Size order breaks because rolling is not predicted

- **Symptom**: early on, a strawberry placed at the upper left of a grape rolls left, leaving strawberry left and grape right and breaking size order. Trying to place a grape just left of a strawberry at the right edge, it touches → gets knocked to the left edge, leaving no place for a dekopon and collapsing, and so on
- **Suspected cause**: the landing is fixed to the drop column, ignoring rolling after a side hit / coasting on the floor / merge drift. Prioritizing an absolute ideal over next to smaller fruits invites gaps and contact accidents
- **Where to look**: `src/policy.py` (`_settle_x`, `_coast_on_floor`, `_wrong_side_roll_penalty`, `_smaller_neighbor_x`)
- **Noted on**: 2026-07-30
- **Status**: fixed — side rolling + floor sliding, penalties for dropping on the big side and knock distance, preferring right next to smaller fruits. Pinned in `tests/test_policy.py`

## Moves were made while the board was still moving

- **Symptom**: even mid-roll / mid-cascade, when held is visible it captures → makes a policy decision. Aims at moving coordinates and misses
- **Suspected cause**: `wait_playable` returned when `ready` even after a settle timeout. Tracker smoothing tends to treat slow rolling as still. `choose_x` ran on observations before settling
- **Where to look**: `src/settle.py`, `src/env.py` (`step`), `src/observe.py` (`raw_fruits`)
- **Noted on**: 2026-07-31
- **Status**: fixed — stillness judged on raw coordinates. Inside `step`, the column is decided on the same observation after settling → aim. Pinned in `tests/test_env.py`

## The view swings too far on edge placements

- **Symptom**: when placing at the right edge, the view swings wastefully through (1) only the view advancing during fine adjustment at the wall (2) a round trip back to center every time. The left edge likewise
- **Where to look**: `src/control.py` (`aim`, `recenter`), `src/env.py`
- **Noted on**: 2026-07-31
- **Status**: fixed — early stop in the edge band + stall when held does not move. `recenter` only pulls back lightly from the edge. Pinned in `tests/test_control.py`

## Cherries packed into the valley between orange and grape

- **Symptom**: early on, drops held=cherry between the orange and grape (a tight valley / shoulder). Fills the firing point and easily breaks size order too
- **Suspected cause**: `_gap_junk_penalty` only looked at floor landings and missed valleys on shoulders
- **Where to look**: `src/policy.py` (`_gap_junk_penalty`)
- **Noted on**: 2026-07-31
- **Status**: fixed — penalized on the floor and in valleys alike. Pinned in `tests/test_policy.py`

## After an edge cherry, strawberries get knocked to the far side

- **Symptom**: move 1 cherry at the right edge, move 2 strawberry dropped "directly on top" (slightly left due to the drop column limit) hits the shoulder, slides to the left edge, giving cherry at the right edge / strawberry at the left edge
- **Suspected cause**: the policy chooses the left neighbor, but aim's early edge stop applied to the whole `EDGE_BAND`. Even for an inner aim near the edge, held stayed at the wall and was treated as "enough", dropping onto the shoulder
- **Where to look**: `src/control.py` (`_edge_close_enough`), `src/policy.py` (neighbor vs directly on top)
- **Noted on**: 2026-07-31
- **Status**: fixed — the early stop applies only right at the wall itself. Pinned in `tests/test_control.py` / `tests/test_policy.py`

## Push-in aims miss

- **Symptom**: it looks like aiming to join same types / push to the edge, but it falls inside and misses
- **Suspected cause**: aiming at a column right at contact tends to go inside due to aim error
- **Where to look**: `src/policy.py` (`PUSH_OUTSET`)
- **Noted on**: 2026-07-31
- **Status**: fixed — the aim column for push-in / restoring is offset outward. The no-shoot gate was withdrawn since frequent `aim_miss` made it weak

## Deferred

- **Training (RL)**: fine to start once the thin bootstrap policy keeps playing for minutes. Per-position placement UTs were dropped (adding concrete heuristics becomes a long-term habit)
- **`10.png` vision**: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than the policy / held

## Policy (bootstrap)

- `src/policy.py` is a thin policy before RL. Only merging, dangerous height, burying, light size order and accident prevention for rolling / knock-aways / gap junk
- Not included: push-in merges, restoring pushes, growing priority, cascade gap opening, forced moves one tier up
- Do not add UTs for concrete procedures. When something breaks, look at accident prevention or the observation side
