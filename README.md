# suika-ai

An AI that watches a Suika Game on screen, chooses a drop column, and plays.

Right now you can play with a **bootstrap policy** (rule-based). There is also a headless simulator and a REINFORCE trainer for a linear policy.

## Requirements

- Windows
- Python 3.11+ recommended

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For development (pytest):

```powershell
pip install -r requirements-dev.txt
```

## Live play

With the game running:

```powershell
python main.py
```

| Key | Action |
|-----|--------|
| `p` | Drop one fruit with the policy |
| `g` | Toggle auto-play (global hotkey; works without focus) |
| `s` | Save a debug frame |
| `Esc` | Quit |

Detection overlay and aim column are shown. Settings live in `config/config.json` (edits apply on the next frame).

## Scripts

```powershell
# Evaluate the thin bootstrap policy in sim
python scripts/eval_bootstrap.py
python scripts/eval_bootstrap.py --episodes 50

# REINFORCE a linear policy in sim (numpy only)
python scripts/train_sim.py
python scripts/train_sim.py --episodes 200 --lr 0.02

# Drop one fruit on the real game (smoke test)
python scripts/play_step.py
python scripts/play_step.py 200

# Run detection on saved images → debug/check/
python scripts/check_detection.py
```

## Tests

```powershell
pytest
```

## Layout

```
main.py                 # live loop (capture → observe → policy → act)
config/config.json      # runtime config
src/
  env.py                # drop → wait → read
  observe.py            # observation (ready / held / next / fruits)
  policy.py             # bootstrap policy (merges, danger height, mishap guards)
  agent.py              # linear policy over 20 discrete columns
  encode.py             # fixed-length observation vector
  reward.py             # survival, merges, max stage, watermelons, death
  sim_env.py            # headless drop sim
  capture.py / control.py / settle.py
  vision/               # board, fruit, held / next detection
scripts/
  eval_bootstrap.py
  train_sim.py
  play_step.py
  check_detection.py
tests/
```

## Notes

Known limits and deferred work around policy / training are in [NOTES.md](NOTES.md).
