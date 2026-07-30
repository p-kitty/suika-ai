import ctypes
import time

import cv2
import numpy as np

from src.capture import capture
from src.config import load
from src.debug_dump import dump
from src.draw import put_text
from src.env import Env
from src.observe import Observation
from src.policy import choose_x
from src.settle import wait_playable
from src.vision.board import draw_frame_debug
from src.vision.held import DROP_HEIGHT
from src.vision.normalized import inverse_warp_matrix, transform_point
from src.window import maximize_window

WINDOW_TITLE = "Suika"
MESSAGE_SECONDS = 3.0

DUMP_KEY = ord("s")
POLICY_KEY = ord("p")
QUIT_KEY = 27
# フォーカスに関係なく auto トグルする (VK_G)。
VK_G = 0x47

# デバッグ表示用の検出周期。毎フレームフル検出すると重いので間引く。
VISION_HZ = 10.0


def main() -> None:
    env = Env()
    message = ""
    message_until = 0.0
    next_auto_dump = 0.0
    next_vision = 0.0
    aim_x: float | None = None
    auto_play = False
    obs = Observation(
        ready=False,
        blocked=False,
        fruits=(),
        held_type=None,
        held_x=None,
        next_type=None,
    )

    maximize_window(WINDOW_TITLE)
    g_was_down = False

    while True:
        frame = capture()
        if frame is None:
            continue

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()

        # G は VRChat 前面でも効くようグローバル検出。押しっぱなしで連打しない。
        g_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_G) & 0x8000)
        g_pressed = g_down and not g_was_down
        g_was_down = g_down

        # キー操作やダンプの直前は最新が欲しい。それ以外は間引く。
        need_vision = (
            now >= next_vision
            or key in (POLICY_KEY, DUMP_KEY)
            or g_pressed
            or auto_play
        )
        if need_vision:
            obs = env.observe(frame)
            next_vision = now + 1.0 / VISION_HZ
        board = env.board

        if obs.ready and obs.held_x is not None and aim_x is None:
            aim_x = obs.held_x

        if g_pressed:
            auto_play = not auto_play
            message = f"auto={'ON' if auto_play else 'off'}"
            message_until = now + MESSAGE_SECONDS
            print(message)

        interval = load().get("debug_dump_interval_sec", 0)
        auto_dump = bool(interval) and board is not None and board.found and now >= next_auto_dump

        if key == DUMP_KEY or auto_dump:
            next_auto_dump = now + max(interval, 1)
            if board is not None:
                message = dump(frame, board)
                message_until = now + MESSAGE_SECONDS
                print(message)

        # p = 方策で 1 手。g で連続自動中なら ready のたびに落とす。
        if key == POLICY_KEY or (auto_play and obs.ready and not obs.blocked):
            if not obs.ready:
                message = "policy: not ready"
            else:
                # 連鎖が止まるまで待ってから列を決める。
                obs = wait_playable(env.observe)
                if obs.blocked or not obs.ready:
                    message = "policy: not settled"
                    auto_play = False if obs.blocked else auto_play
                    frame, obs, board = _refresh(env, frame, obs)
                else:
                    target = choose_x(obs)
                    aim_x = target
                    result = env.step(target)
                    message = f"auto x={target:.0f} -> {result.info}"
                    aim_x = result.observation.held_x
                    obs = result.observation
                    frame, obs, board = _refresh(env, frame, obs)
                    print(message)
                    if result.done:
                        auto_play = False
                        message = f"{message} (stop)"
            message_until = now + MESSAGE_SECONDS

        output = frame.copy()
        if board is not None:
            output = draw_frame_debug(frame, board)
            if aim_x is not None and board.corners is not None and obs.ready:
                _draw_aim(output, board.corners, aim_x)

        mode = "AUTO" if auto_play else "LIVE"
        hint = f"{mode}  p: policy  g: auto(global)  s: save"
        put_text(output, f"aim x={aim_x:.0f}" if aim_x is not None else "aim —", (8, 128), (0, 255, 255))
        put_text(
            output,
            message if now < message_until else hint,
            (8, output.shape[0] - 12),
            (255, 255, 255),
            scale=0.5,
        )
        cv2.imshow(WINDOW_TITLE, output)

        if key == QUIT_KEY:
            break

    cv2.destroyAllWindows()


def _refresh(env: Env, frame: np.ndarray, fallback: Observation):
    """操作後に最新フレームで観測し直す。取れなければ直前のものを使う。"""
    fresh = capture()
    if fresh is None:
        return frame, fallback, env.board
    obs = env.observe(fresh)
    return fresh, obs, env.board


def _draw_aim(frame, corners, x: float) -> None:
    matrix = inverse_warp_matrix(corners)
    top = transform_point(matrix, x, -DROP_HEIGHT)
    bottom = transform_point(matrix, x, 40)
    cv2.line(frame, top, bottom, (0, 255, 255), 2)
    cv2.circle(frame, top, 6, (0, 255, 255), -1)


if __name__ == "__main__":
    main()
