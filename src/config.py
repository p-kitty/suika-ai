import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.json"


def load():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)