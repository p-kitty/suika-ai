# Known issues and things to fix later

## Deferred

- **`10.png` vision**: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than policy / held

## Policy (bootstrap)

- `src/policy.py` is a thin policy before RL. Only merging, dangerous height, burying, light size order and accident prevention for rolling / knock-aways / gap junk
- Not included: push-in merges, restoring pushes, growing priority, cascade gap opening, forced moves one tier up
- Do not add UTs for concrete procedures. When something breaks, look at accident prevention or the observation side

## Training

- `src/reward.py`: survival, merges, stage, more watermelons / reaching a double / clear. win (+WIN_BONUS) on a double clear. Death penalty. No maintenance bonus
- `src/encode.py`: fixed-length observation vector
- `src/sim_env.py`: headless drop sim (`policy.simulate_drop`)
- Evaluation: `python scripts/eval_policy.py` (`--policy bootstrap|learned`)
- Training: `python scripts/train_sim.py` (collect → offline BC. The default max-steps=100 is a cap, not the losing line)
- `src/agent.py`: MLP with 32 discrete column bins / hidden 128 (old 20/64 npz files need retraining)
- Live play: `python main.py` (defaults to learned if an npz exists. `L` toggles bootstrap, `--policy bootstrap`)

## Planned: RL (REINFORCE)

- Still too early. Plain REINFORCE easily breaks things when BC is shallow (confirmed in the past)
- Condition for adding it: `match` fairly high (roughly 60–70%+) and `student_r` close to bootstrap (~70)
- How: only a short fine-tune after BC finishes (e.g. `--episodes 50 --lr 0.002`). Off by default
- Until then, thickening BC (collection size, epochs) comes first
