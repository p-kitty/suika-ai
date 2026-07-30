import ctypes
import time

import cv2
import numpy as np

from src.capture import CAPTURE_FPS, capture
from src.config import load
from src.debug_dump import dump
from src.draw import mode_badge, put_text
from src.env import Env
from src.observe import Observation
from src.policy import choose_x
from src.vision.board import draw_frame_debug
from src.vision.held import DROP_HEIGHT
from src.vision.normalized import inverse_warp_matrix, transform_point
from src.window import maximize_window

WINDOW_TITLE = "Suika"
MESSAGE_SECONDS = 3.0

DUMP_KEY = ord("s")
STEP_KEY = ord(" ")
QUIT_KEY = 27
# Toggle auto regardless of focus (VK_G).
VK_G = 0x47

# Preview while waiting. Only the video runs; stale detection circles are not drawn.
PUMP_HZ = float(CAPTURE_FPS)


def main() -> None:
    env = Env()
    message = ""
    message_until = 0.0
    next_auto_dump = 0.0
    next_pump = 0.0
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
    frame: np.ndarray | None = None

    def poll_g_toggle() -> bool:
        """Toggle auto on the rising edge of G. True when pressed."""
        nonlocal auto_play, g_was_down, message, message_until
        down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_G) & 0x8000)
        pressed = down and not g_was_down
        g_was_down = down
        if not pressed:
            return False
        auto_play = not auto_play
        message = f"auto={'ON' if auto_play else 'off'}"
        message_until = time.monotonic() + MESSAGE_SECONDS
        print(message)
        return True

    def pump_ui() -> None:
        """Keep only the video running while waiting for settle / aim. Stale detection circles are not drawn."""
        nonlocal frame, next_pump
        now = time.monotonic()
        if now < next_pump:
            cv2.waitKey(1)
            return
        next_pump = now + 1.0 / PUMP_HZ
        fresh = capture()
        if fresh is not None:
            frame = fresh
        if frame is None:
            return
        # No detection overlay. Drawing the previous board on a new frame stutters.
        output = frame.copy()
        mode_badge(output, auto_play)
        put_text(
            output,
            message if now < message_until else "settling...",
            (8, output.shape[0] - 12),
            (255, 255, 255),
            scale=0.5,
        )
        cv2.imshow(WINDOW_TITLE, output)
        cv2.waitKey(1)

    def should_abort() -> bool:
        """For wait loops during auto. Abort when switched off with G."""
        poll_g_toggle()
        pump_ui()
        return not auto_play

    while True:
        fresh = capture()
        if fresh is not None:
            frame = fresh
        if frame is None:
            # Only watch keys until the first frame arrives.
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                break
            continue

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()

        # G is detected globally so it works even with VRChat in front. Holding does not repeat.
        # While waiting in step / settle, the same detection runs via should_abort.
        poll_g_toggle()

        # Detect only when a new capture arrives. Keeps video and overlay on the same cycle.
        if fresh is not None or key in (STEP_KEY, DUMP_KEY):
            obs = env.observe(frame)
        board = env.board

        if obs.ready and obs.held_x is not None and aim_x is None:
            aim_x = obs.held_x

        interval = load().get("debug_dump_interval_sec", 0)
        auto_dump = bool(interval) and board is not None and board.found and now >= next_auto_dump

        if key == DUMP_KEY or auto_dump:
            next_auto_dump = now + max(interval, 1)
            if board is not None:
                message = dump(frame, board)
                message_until = now + MESSAGE_SECONDS
                print(message)

        # Space = one move. During continuous auto with g, drop on every ready.
        # Draw AUTO/LIVE before entering the wait so it shows.
        from_auto = auto_play and key != STEP_KEY
        if key == STEP_KEY or (auto_play and obs.ready and not obs.blocked):
            message = "settling..."
            message_until = now + MESSAGE_SECONDS
            _show(frame, board, obs, aim_x, auto_play, message, message_until, now)

            def abort() -> bool:
                # Keep the preview running while waiting. On auto, G can abort it.
                if from_auto:
                    return should_abort()
                pump_ui()
                return False

            if not obs.ready:
                message = "step: not ready"
                print(message)
            else:
                # Confirm settle → decide the column on the same observation → aim. Do not read a moving board.
                result = env.step(abort=abort, choose=choose_x)
                if from_auto and not auto_play:
                    message = "auto=off"
                    frame, obs, board = _refresh(env, frame, obs)
                elif result.info == "not settled":
                    message = "step: not settled"
                    print(message)
                    frame, obs, board = _refresh(env, frame, obs)
                else:
                    target = result.target_x
                    aim_x = target
                    message = (
                        f"auto x={target:.0f} -> {result.info}"
                        if target is not None
                        else f"auto -> {result.info}"
                    )
                    aim_x = result.observation.held_x
                    obs = result.observation
                    frame, obs, board = _refresh(env, frame, obs)
                    print(message)
                    if from_auto and not auto_play:
                        message = f"{message} (stop)"
                    elif result.done:
                        auto_play = False
                        message = f"{message} (stop)"
            message_until = time.monotonic() + MESSAGE_SECONDS

        _show(frame, board, obs, aim_x, auto_play, message, message_until, now)

        if key == QUIT_KEY:
            break

    cv2.destroyAllWindows()


def _show(
    frame: np.ndarray,
    board,
    obs: Observation,
    aim_x: float | None,
    auto_play: bool,
    message: str,
    message_until: float,
    now: float,
) -> None:
    output = frame.copy()
    if board is not None:
        output = draw_frame_debug(frame, board)
        if aim_x is not None and board.corners is not None and obs.ready:
            _draw_aim(output, board.corners, aim_x)

    mode_badge(output, auto_play)
    hint = "Space: step  g: auto on/off  s: save"
    put_text(output, f"aim x={aim_x:.0f}" if aim_x is not None else "aim —", (8, 128), (0, 255, 255))
    put_text(
        output,
        message if now < message_until else hint,
        (8, output.shape[0] - 12),
        (255, 255, 255),
        scale=0.5,
    )
    cv2.imshow(WINDOW_TITLE, output)
    cv2.waitKey(1)


def _refresh(env: Env, frame: np.ndarray, fallback: Observation):
    """Re-observe on the latest frame after the action. Use the previous one if none is available."""
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
