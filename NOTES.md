# Known issues and things to fix later

## HELD / NEXT disappear and play stalls (near the top edge of the board)

- **Symptom**: when a pineapple sits where it almost sticks out the top, HELD and NEXT stop being detected and auto play stalls
- **How to reproduce**: positions with a big fruit (pineapple and so on) near the top edge of the board
- **Where to look**: `src/vision/held.py`, `src/vision/next.py`, masks / ROI of fruits sticking out above
- **Noted on**: 2026-07-30
- **Status**: not started
