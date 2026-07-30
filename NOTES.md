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
- **Status**: fixed — when held/next are the same type, prefer "on top of" the growing target. Pinned by position UTs in `tests/test_policy.py`

## Deferred

- **Training (RL)**: while holes in observation and policy remain it becomes noise. Only after position UTs reduce misplacements and play runs for minutes without stopping
- **`10.png` vision**: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than policy / held
