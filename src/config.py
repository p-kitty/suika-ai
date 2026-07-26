import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.json"

Config = dict[str, Any]

_cache: Config | None = None
_cache_mtime: float | None = None


def load() -> Config:
    """設定を読む。編集したら次のフレームから反映される。"""
    global _cache, _cache_mtime

    mtime = CONFIG_PATH.stat().st_mtime
    if _cache is None or mtime != _cache_mtime:
        _cache = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        _cache_mtime = mtime

    return _cache
