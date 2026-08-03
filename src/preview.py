from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.draw import mode_badge, put_text
from src.observe import Observation
from src.vision.board import draw_frame_debug
from src.vision.held import DROP_HEIGHT
from src.vision.normalized import inverse_warp_matrix, transform_point

HINT = "p: step  g: auto  l: policy  s: save"


@dataclass
class PreviewState:
    aim_x: float | None
    auto_play: bool
    policy_name: str
    message: str
    message_until: float
    now: float


def draw_aim_line(frame: np.ndarray, corners, x: float) -> None:
    matrix = inverse_warp_matrix(corners)
    top = transform_point(matrix, x, -DROP_HEIGHT)
    bottom = transform_point(matrix, x, 40)
    cv2.line(frame, top, bottom, (0, 255, 255), 2)
    cv2.circle(frame, top, 6, (0, 255, 255), -1)


def render_preview(
    frame: np.ndarray,
    *,
    board,
    obs: Observation,
    state: PreviewState,
    debug_board: bool,
    window_title: str,
) -> None:
    if debug_board and board is not None:
        output = draw_frame_debug(frame, board)
        # 検出オーバーレイと揃えるので、読めているときだけ狙い線を出す。
        aim = state.aim_x if obs.ready else None
    else:
        output = frame.copy()
        # 待ち中は古い検出を載せない代わりに、狙い線だけ残す。
        aim = state.aim_x

    if aim is not None and board is not None and board.corners is not None:
        draw_aim_line(output, board.corners, aim)

    mode_badge(output, state.auto_play)
    put_text(
        output,
        f"aim x={state.aim_x:.0f}" if state.aim_x is not None else "aim —",
        (8, 128),
        (0, 255, 255),
    )
    put_text(
        output,
        f"policy={state.policy_name}",
        (8, 152),
        (0, 220, 255),
        scale=0.5,
    )
    if state.now < state.message_until:
        footer = state.message
    elif debug_board:
        footer = HINT
    else:
        footer = "settling..."
    put_text(
        output,
        footer,
        (8, output.shape[0] - 12),
        (255, 255, 255),
        scale=0.5,
    )
    cv2.imshow(window_title, output)
    cv2.waitKey(1)
