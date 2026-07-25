from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DUMP_DIR = Path(__file__).resolve().parents[1] / "debug"


def dump(frame: np.ndarray, result) -> str:
    """デバッグ描画を含まない生の画像を保存する。

    表示用のウィンドウをスクリーンショットすると自分が描いた円や枠が
    写り込み、しきい値の調整に使えないため。
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
    """cv2.imwrite は非 ASCII のパスだと黙って失敗するので、
    エンコードした結果を自分で書き出す。"""
    success, buffer = cv2.imencode(path.suffix, image)
    if not success:
        return False

    path.write_bytes(buffer.tobytes())
    return True
