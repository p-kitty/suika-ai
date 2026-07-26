from pathlib import Path

import cv2
import numpy as np

# cv2.imread と cv2.imwrite は非 ASCII のパスだと黙って失敗するので、
# エンコード済みのバイト列を自分で読み書きする。


def read(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None

    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def write(path: Path, image: np.ndarray) -> bool:
    success, buffer = cv2.imencode(path.suffix, image)
    if not success:
        return False

    path.write_bytes(buffer.tobytes())
    return True
