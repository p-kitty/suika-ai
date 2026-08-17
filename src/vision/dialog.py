from functools import cache
from pathlib import Path

import cv2
import numpy as np

from ..util.imagefile import read

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "continue_button.png"

# ダイアログが出ていないフレームでの一致度は手元の 8 枚すべてで 0.45 以下、
# 出ているフレームは 1.0 だった。間は広いので余裕をもって切る。
MATCH_THRESHOLD = 0.65


def is_blocked(board: np.ndarray) -> bool:
    """盤面がダイアログで覆われているか。

    Game Over や New High Score が出ている間、盤面はダイアログの下に隠れる。
    読み取っても意味がないどころか、ダイアログ自体をフルーツとして拾うので、
    手を出さないフレームとして弾く必要がある。

    どちらのダイアログにも同じ Continue ボタンがあるので、それを探す。
    warp 後の盤面なら見かけの大きさが揃うため、縮尺を気にせず照合できる。
    """
    template = _template()
    if template is None:
        return False

    if board.shape[0] < template.shape[0] or board.shape[1] < template.shape[1]:
        return False

    score = cv2.matchTemplate(board, template, cv2.TM_CCOEFF_NORMED).max()

    return float(score) >= MATCH_THRESHOLD


@cache
def _template() -> np.ndarray | None:
    return read(TEMPLATE_PATH)
