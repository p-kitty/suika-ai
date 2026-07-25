import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.json"

_cache = None
_cache_mtime = None


def load():
    global _cache, _cache_mtime

    mtime = CONFIG_PATH.stat().st_mtime
    if _cache is None or mtime != _cache_mtime:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
        _cache_mtime = mtime

    return _cache


def save(cfg):
    global _cache, _cache_mtime

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    _cache = None
    _cache_mtime = None
