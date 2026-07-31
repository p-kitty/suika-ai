# Known issues and things to fix later

## Deferred

- **`10.png` vision**: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than policy / held
- **`GAP_JUNK_PENALTY`**: gap filling can be both an accident and the seed of a cascade. Judging only by score results is stronger than forbidding it by hand. Remove later
- **Training episode length**: raise `max_steps` and lower `episodes` (fewer, longer games)

## Policy (bootstrap)

- `src/policy.py` is a thin policy before RL. Only merging, dangerous height, burying, light size order and accident prevention for rolling / knock-aways / gap junk
- Moves are scored as `eval = score - penalties`. The only bonus is the real game's score; stacking, accidents and burying are penalties
- Burying is the main penalty. Moves that block a same-type pair waiting to merge with a bigger fruit of another type, directly above or on the shoulder, are heavily penalized
- Aiming at the center of a different type (`FOREIGN_AIM`) and excess same type (`EXCESS_SAME` = 20 per excess fruit) suppress breaking and delayed merging. Stacking a different type in valleys or on shoulders is not forbidden
- Not included: push-in merges, restoring pushes, growing priority, cascade gap opening, forced moves one tier up
- Do not add UTs for concrete procedures. When something breaks, look at accident prevention or the observation side

## Training

- **score / eval**: `score` is the real game's merge score (1-65, no penalties), `penalties` are the penalties for accidents and bad moves, `eval = score - penalties`. Policy move selection and the student's quality use eval; **the RL reward stays score** (dense penalties are not rewards)
- `src/reward.py`: `merge_score(merge_types)` gives only merge points identical to the real game (cherry→0 … watermelon 55, double clear 65). No survival bonus or death penalty. Episodes end as before (losing line / double clear)
- `src/encode.py`: fixed-length observation vector
- `src/sim_env.py`: headless drop sim (`policy.simulate_drop`). `SimStep` has `score` and `eval_score`
- Evaluation: `python scripts/eval_policy.py` (`--policy bootstrap|learned`. `--workers` default = logical cores/2)
- Training: `python scripts/train_sim.py` (collect → offline BC. The default max-steps=100 is a cap, not the losing line). Logs show score and eval side by side, and best is chosen by eval
- Teacher collection runs in parallel with `ProcessPool` (default workers=logical cores/2; 8 on a 9700X; `--workers 1` for serial)
- `src/agent.py`: MLP with 32 discrete column bins / hidden 128 (old 20/64 npz files need retraining)
- Live play: `python main.py` (defaults to learned if an npz exists. `L` toggles bootstrap, `--policy bootstrap`)

## Planned: RL (REINFORCE)

- Still too early. Plain REINFORCE easily breaks things when BC is shallow (confirmed in the past)
- Condition for adding it: `match` fairly high (roughly 60–70%+) and the student's `score` / `eval` close to bootstrap
- How: only a short fine-tune after BC finishes (e.g. `--episodes 50 --lr 0.002`). Off by default
- Until then, thickening BC (collection size, epochs) comes first
