from datetime import datetime
from pathlib import Path

import numpy as np

from .imagefile import write
from .vision.board import NORMALIZED_WIDTH, BoardResult
from .vision.fruits import fruit_mask

DUMP_DIR = Path(__file__).resolve().parents[1] / "debug"


def dump(frame: np.ndarray, result: BoardResult) -> str:
    """Save raw images that contain no debug drawing.

    Screenshotting the display window captures the circles and frames it drew itself,
    which makes them useless for tuning thresholds.
    """
    DUMP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")

    images = {"frame": frame}
    if result.normalized is not None:
        images["board"] = result.normalized
        images["mask"] = fruit_mask(result.normalized)

    saved = []
    for name, image in images.items():
        path = DUMP_DIR / f"{stamp}_{name}.png"
        if write(path, image):
            saved.append(path.name)

    if not saved:
        return "dump failed"

    details = " ".join(
        f"{fruit.name}={fruit.radius / NORMALIZED_WIDTH:.3f}"
        for fruit in result.fruits or []
    )
    return f"saved {', '.join(saved)} -> {DUMP_DIR}" + (f" | {details}" if details else "")
