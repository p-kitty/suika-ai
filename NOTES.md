# Known issues and things to fix later

## Deferred

- **`10.png` vision**: similar-color mask fusion + a strawberry outside the frame. Needs a redesign of the cropping; worse value for effort than the policy / held

## Policy (bootstrap)

- `src/policy.py` is a thin policy before RL. Only merging, dangerous height, burying, light size order and accident prevention for rolling / knock-aways / gap junk
- Not included: push-in merges, restoring pushes, growing priority, cascade gap opening, forced moves one tier up
- Do not add UTs for concrete procedures. When something breaks, look at accident prevention or the observation side

## Training

- `src/reward.py`: survival, merges, max stage, watermelon / double watermelon, death
- `src/encode.py`: fixed-length observation vector
- `src/sim_env.py`: headless drop sim (`policy.simulate_drop`)
- Evaluation: `python scripts/eval_bootstrap.py`
- Training: `python scripts/train_sim.py` (BC+replay with bootstrap as the teacher. MLP. RL off by default)
- `src/agent.py`: MLP policy over 20 discrete column bins
