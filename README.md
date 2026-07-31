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
| `p` | Drop one fruit (one step; global hotkey) |
| `g` | Toggle auto-play (global hotkey; works without focus) |
| `s` | Save a debug frame |
| `Esc` | Quit |

Detection overlay and aim column are shown. Settings live in `config/config.json` (edits apply on the next frame).

## Scripts

```powershell
# Evaluate bootstrap / learned policy in sim
python scripts/eval_policy.py --policy bootstrap
python scripts/eval_policy.py --policy learned --episodes 20

# Train offline BC (+ optional REINFORCE) in sim
python scripts/train_sim.py
python scripts/train_sim.py --bc-episodes 100 --episodes 50 --lr 0.002

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
  reward.py             # game-identical merge scores only
  sim_env.py            # headless drop sim
  capture.py / control.py / settle.py
  vision/               # board, fruit, held / next detection
scripts/
  eval_policy.py
  train_sim.py
  check_detection.py
tests/
```

## Notes

Known limits and deferred work around policy / training are in [NOTES.md](NOTES.md).
