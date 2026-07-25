from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DUMP_DIR = Path(__file__).resolve().parents[1] / "debug"


def dump(frame: np.ndarray, result) -> str:
    """Save raw images that contain no debug drawing.

    Screenshotting the display window captures the circles and frames it drew itself,
    which makes them useless for tuning thresholds.
    """
    from .vision.fruits import _fruit_mask

    DUMP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")

    images = {"frame": frame}
    if result.normalized is not None:
        images["board"] = result.normalized
        images["mask"] = _fruit_mask(result.normalized)

    saved = []
    for name, image in images.items():
        path = DUMP_DIR / f"{stamp}_{name}.png"
        if _write(path, image):
            saved.append(path.name)

    if not saved:
        return "dump failed"

    details = " ".join(
        f"{fruit.name}={fruit.radius / 400:.3f}" for fruit in result.fruits or []
    )
    return f"saved {', '.join(saved)} -> {DUMP_DIR}" + (f" | {details}" if details else "")


def _write(path: Path, image: np.ndarray) -> bool:
    """cv2.imwrite fails silently on non-ASCII paths,
    so the encoded result is written out by hand."""
    success, buffer = cv2.imencode(path.suffix, image)
    if not success:
        return False

    path.write_bytes(buffer.tobytes())
    return True
