# Known issues and things to fix later

## HELD / NEXT disappear and play stalls (near the top edge of the board)

- **Symptom**: when a pineapple sits where it almost sticks out the top, HELD and NEXT stop being detected and auto play stalls
- **How to reproduce**: positions with a big fruit (pineapple and so on) near the top edge of the board
- **Where to look**: `src/vision/held.py`, `src/vision/next.py`, masks / ROI of fruits sticking out above
- **Noted on**: 2026-07-30
- **Status**: not started

## Placement does not account for NEXT / HELD

- **Symptom**: e.g. in a position wanting to grow a grape, with NEXT and HELD both strawberries, it does not put them on the grape
- **Suspected cause**: the policy / evaluation centers on "the fruit held now" and does not sufficiently consider cascades and growing based on NEXT and HELD
- **Where to look**: `src/policy.py`, placement scoring / candidate evaluation
- **Noted on**: 2026-07-30
- **Status**: not started
