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

- **Symptom**: even mid-roll / mid-cascade, when held is visible it captures → makes a policy decision
- **Suspected cause**: `wait_playable` returned when `ready` even after a settle timeout. On top of that, Tracker smoothing tends to treat slow rolling as still
- **Where to look**: `src/settle.py` (`wait_settled`, `wait_playable`)
- **Noted on**: 2026-07-31
- **Status**: fixed — playable only when stillness is confirmed. Thresholds tightened too. A timeout does not stop auto. Pinned in `tests/test_env.py`

## The view swings too far on edge placements

- **Symptom**: when placing at the right edge, the view swings wastefully through (1) only the view advancing during fine adjustment at the wall (2) a round trip back to center every time. The left edge likewise
- **Where to look**: `src/control.py` (`aim`, `recenter`), `src/env.py`
- **Noted on**: 2026-07-31
- **Status**: fixed — early stop in the edge band + stall when held does not move. `recenter` only pulls back lightly from the edge. Pinned in `tests/test_control.py`

## Push-in aims miss

- **Symptom**: it looks like aiming to join same types / push to the edge, but it falls inside and misses
- **Suspected cause**: aiming at a column right at contact tends to go inside due to aim error
- **Where to look**: `src/policy.py` (`PUSH_OUTSET`)
- **Noted on**: 2026-07-31
- **Status**: fixed — the aim column for push-in / restoring is offset outward. The no-shoot gate was withdrawn since frequent `aim_miss` made it weak

## Deferred

- **Training (RL)**: while holes in observation and policy remain it becomes noise. Only after position UTs reduce misplacements and play runs for minutes without stopping
- **`10.png` vision**: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than the policy / held

## Placement UT

- For boards, look at `screenshots/doko*.png`. No fruit coordinate lists are kept
- `tests/expected_drops.py`: held/next (human ground truth) and `expect_x=(lo,hi)` (drop column)
- `tests/test_drops.py` runs localize → choose_x end to end
  - held/next: compare detection results against expected (do not pass expected straight to the policy)
  - expect_x: unique by column even with multiple same types. Name references (`on grape`) are not used
- Unsolvable positions get a strict xfail in `KNOWN_DROP_FAILURES` (for policy mistakes; broken detection stays red)
- **Teach only one move**: the second move depends on the next held (the current next), so it is not written
- Priority guide: same-type merge → if held/next are the same type, growing one tier up (the side where it lines up / push to the big side; not directly above the center of a different type) → pushing a broken size order back to the edge → a column that does not break size order. Placements that collapse, such as on a melon's shoulder, are not allowed
- **Directly above the center of a different type**: penalized rather than rewarded. If the lining-up side is open go there, if blocked push to the big side (`_foreign_center_penalty`, `_large_side_x`). Pinned with doko3/8/9
- **Push-in**: with a held of a different type, hitting the outside of a nearby same-type pair to join them is rewarded (`_push_merge_bonus`). Pinned with doko2
- **Restoring push**: when size order is locally inverted, push from the small side's outside toward the big side's edge with a held of dekopon or bigger (`_restore_order_bonus`). The size-order penalty was strengthened too. Pinned by position UTs
- **Gap junk**: do not stuff a small fruit into the floor between fruits 2 or more stages bigger than itself (`_gap_junk_penalty`). Pinned by position UTs
- Main holes among the remaining xfails: sliding next to a merge column (doko12), gap sliding and melon shoulders (doko4/10/11), doko5/7
