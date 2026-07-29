import time

import cv2

from src.capture import capture
from src.config import load
from src.debug_dump import dump
from src.draw import put_text
from src.env import Env
from src.observe import Observation, clamp_drop_x
from src.vision.board import draw_frame_debug
from src.vision.held import DROP_HEIGHT
from src.vision.normalized import inverse_warp_matrix, transform_point
from src.window import maximize_window

WINDOW_TITLE = "Suika"
MESSAGE_SECONDS = 3.0

DUMP_KEY = ord("s")
DROP_KEY = ord(" ")
LEFT_KEY = ord("a")
RIGHT_KEY = ord("d")
CENTER_KEY = ord("c")
QUIT_KEY = 27

# 手動で狙うときの左右の刻み (正規化座標)。
NUDGE = 12.0
# デバッグ表示用の検出周期。毎フレームフル検出すると重いので間引く。
VISION_HZ = 10.0


def main() -> None:
    env = Env()
    message = ""
    message_until = 0.0
    next_auto_dump = 0.0
    next_vision = 0.0
    # None のあいだは落下待ちの今の列に合わせる。a/d で動かしたら固定。
    aim_x: float | None = None
    obs = Observation(
        ready=False,
        blocked=False,
        fruits=(),
        held_type=None,
        held_x=None,
        next_type=None,
    )

    maximize_window(WINDOW_TITLE)

    while True:
        frame = capture()
        if frame is None:
            continue

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()

        # キー操作やダンプの直前は最新が欲しい。それ以外は間引く。
        need_vision = (
            now >= next_vision
            or key in (LEFT_KEY, RIGHT_KEY, CENTER_KEY, DROP_KEY, DUMP_KEY)
        )
        if need_vision:
            obs = env.observe(frame)
            next_vision = now + 1.0 / VISION_HZ
        board = env.board

        if obs.ready and obs.held_x is not None:
            if aim_x is None:
                aim_x = obs.held_x
            aim_x = clamp_drop_x(aim_x, obs.held_type)

        if key == LEFT_KEY and aim_x is not None:
            aim_x = clamp_drop_x(aim_x - NUDGE, obs.held_type)
        elif key == RIGHT_KEY and aim_x is not None:
            aim_x = clamp_drop_x(aim_x + NUDGE, obs.held_type)
        elif key == CENTER_KEY and obs.held_x is not None:
            aim_x = obs.held_x

        interval = load().get("debug_dump_interval_sec", 0)
        auto = bool(interval) and board is not None and board.found and now >= next_auto_dump

        if key == DUMP_KEY or auto:
            next_auto_dump = now + max(interval, 1)
            if board is not None:
                message = dump(frame, board)
                message_until = now + MESSAGE_SECONDS
                print(message)

        if key == DROP_KEY:
            if not obs.ready or aim_x is None:
                message = "drop: not ready"
            else:
                result = env.step(aim_x)
                message = f"drop x={aim_x:.0f} -> {result.info}"
                aim_x = result.observation.held_x
                obs = result.observation
                print(message)
            message_until = now + MESSAGE_SECONDS

        output = frame.copy()
        if board is not None:
            output = draw_frame_debug(frame, board)
            if aim_x is not None and board.corners is not None and obs.ready:
                _draw_aim(output, board.corners, aim_x)

        mode = "LIVE" if not env.dry_run else "dry-run"
        hint = f"{mode}  a/d: aim  space: drop  c: held  s: save"
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


def _draw_aim(frame, corners, x: float) -> None:
    matrix = inverse_warp_matrix(corners)
    top = transform_point(matrix, x, -DROP_HEIGHT)
    bottom = transform_point(matrix, x, 40)
    cv2.line(frame, top, bottom, (0, 255, 255), 2)
    cv2.circle(frame, top, 6, (0, 255, 255), -1)


if __name__ == "__main__":
    main()
