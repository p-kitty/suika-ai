from functools import cache
from pathlib import Path

import cv2
import numpy as np

from ..imagefile import read

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "continue_button.png"

# On frames without a dialog the match score was 0.45 or less on all 8 local images,
# and 1.0 on frames with one. The gap is wide, so the cut has plenty of margin.
MATCH_THRESHOLD = 0.65


def is_blocked(board: np.ndarray) -> bool:
    """Whether the board is covered by a dialog.

    While Game Over or New High Score is shown, the board is hidden under the dialog.
    Reading it is not only meaningless but picks up the dialog itself as fruit,
    so such frames must be rejected as ones not to act on.

    Both dialogs have the same Continue button, so that is what is searched for.
    A warped board has a consistent apparent size, so matching works without worrying about scale.
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
