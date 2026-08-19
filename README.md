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

For development (pytest / basedpyright / ruff / vulture):

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

# Screen a weight change before spending hours on an A/B
python scripts/band_escape.py

# Trace one game and count the fruits that never merge
python scripts/fossils.py --seed 642746

# Measure the real game's fall: constant speed or accelerating?
# (needs the game on screen; drop one fruit on an empty board)
python scripts/measure_fall_speed.py
```

## Tests

```powershell
pytest
basedpyright
# 未使用 import を自動削除
ruff check --fix
# 未参照の関数・定数（誤検知あり。消す前に確認）
vulture
```

## Layout

```
main.py                 # live loop (capture → observe → policy → act)
config/config.json      # runtime config
src/
  observe.py            # observation (ready / held / next / fruits)
  policy.py             # bootstrap policy (merges, danger height, mishap guards)
  penalties.py          # board penalty rules the policy scores against
  ladder.py             # ladder detection (detection only; not wired into play)
  reward.py             # game-identical merge scores only
  vision/               # board, fruit, held / next detection
  game/                 # live game I/O: capture, control, settle, tracker, env
  sim/                  # headless sim: sim_physics (pymunk), sim_env
  training/             # agent, observation encoding, BC + REINFORCE
  viz/                  # drawing helpers, overlay preview, debug frame dump
  util/                 # config, image I/O, worker count, A/B statistics
scripts/
  eval_policy.py
  train_sim.py
  check_detection.py
  compare_policy.py     # A/B two policy variants on the same seeds
  analyze_ab.py         # pick a proxy metric from a compare_policy dump
  band_escape.py        # pre-A/B screen: does a weight change escape the tied band?
  measure_fall_speed.py # calibrate the sim's GRAVITY against the real game
  fossils.py            # diagnosis: which fruits never merge, and what covers them
  view_sim.py           # watch the sim board (mouse to drop, g for auto-play)
tests/                  # mirrors src/ (game/, sim/, training/, util/, vision/)
```

## Notes

Known limits and deferred work around policy / training are in [NOTES.md](NOTES.md).
Conventions and workflow for working in this repo (human or agent) are in [AGENTS.md](AGENTS.md).

## License

[MIT](LICENSE)
