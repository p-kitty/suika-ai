from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.draw import FRUIT_BGR, mode_badge, put_text
from src.observe import Observation, clamp_drop_x
from src.policy import drop_scores
from src.reward import GAME_OVER_Y
from src.vision.board import draw_frame_debug
from src.vision.held import DROP_HEIGHT
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH, inverse_warp_matrix, transform_point
from src.vision.state import Fruit

HINT = "p: step  g: auto  l: policy  s: save"

# 右上の「落としたらこうなる」ミニ盤。
PRED_SCALE = 0.65
PRED_MARGIN = 12
PRED_TOP = 64  # AUTO/LIVE バッジの下に来るぶん空ける

# simulate_drop は毎手 1 回で十分重い。render_preview は毎フレーム (最大
# CAPTURE_FPS 以上の頻度) 呼ばれるので、盤・狙いが変わらない限り使い回す。
_pred_cache: dict[str, object] = {"key": None, "after": None, "merges": 0}


def _predicted_after_drop(
    obs: Observation, aim_x: float | None
) -> tuple[list[Fruit], int] | None:
    if not obs.ready or obs.held_type is None or aim_x is None:
        return None
    x = clamp_drop_x(aim_x, obs.held_type)
    key = (obs.fruits, obs.held_type, obs.next_type, round(x))
    if _pred_cache["key"] != key:
        _, _, _, after, merges = drop_scores(
            obs.fruits, obs.held_type, x, next_type=obs.next_type
        )
        _pred_cache["key"] = key
        _pred_cache["after"] = after
        _pred_cache["merges"] = merges
    return _pred_cache["after"], _pred_cache["merges"]  # type: ignore[return-value]


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
    _draw_prediction_panel(output, obs, aim)
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


def _draw_prediction_panel(
    output: np.ndarray,
    obs: Observation,
    aim_x: float | None,
) -> None:
    """右上に「いま盤にあるフルーツ」と「そこへ落としたらこうなる」をミニ表示する。"""
    predicted = _predicted_after_drop(obs, aim_x)
    if predicted is None:
        return
    after, merges = predicted

    panel_w = max(1, int(round(NORMALIZED_WIDTH * PRED_SCALE)))
    panel_h = max(1, int(round(NORMALIZED_HEIGHT * PRED_SCALE)))
    x0 = output.shape[1] - panel_w - PRED_MARGIN
    y0 = PRED_TOP
    if x0 < 0 or y0 + panel_h + 20 > output.shape[0]:
        return

    # mode_badge 同様、フル画面コピーの alpha blend は毎フレーム呼ばれるには
    # 重いので使わない。塗りつぶし矩形を直接描く。
    cv2.rectangle(output, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    cv2.rectangle(output, (x0, y0), (x0 + panel_w, y0 + panel_h), (110, 110, 110), 1)

    danger_y = y0 + int(round(GAME_OVER_Y * PRED_SCALE))
    cv2.line(output, (x0, danger_y), (x0 + panel_w, danger_y), (40, 40, 160), 1)

    for fruit in after:
        cx = x0 + int(round(fruit.x * PRED_SCALE))
        cy = y0 + int(round(fruit.y * PRED_SCALE))
        r = max(1, int(round(fruit.radius * PRED_SCALE)))
        bgr = FRUIT_BGR[fruit.type]
        cv2.circle(output, (cx, cy), r, bgr, -1)
        cv2.circle(output, (cx, cy), r, (240, 240, 240), 2)

    put_text(output, "AFTER DROP", (x0, y0 - 8), (210, 210, 210), scale=0.6, thickness=1)
    tail = f"merges={merges}" if merges else "no merge"
    put_text(
        output,
        tail,
        (x0, y0 + panel_h + 20),
        (180, 255, 180) if merges else (180, 180, 180),
        scale=0.55,
        thickness=1,
    )
