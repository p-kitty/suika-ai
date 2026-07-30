import ctypes
import time

import cv2
import numpy as np

from src.capture import capture
from src.config import load
from src.debug_dump import dump
from src.draw import mode_badge, put_text
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
# Toggle auto regardless of focus (VK_G).
VK_G = 0x47

# Detection interval for the debug display. Full detection every frame is heavy, so it is thinned.
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

    def should_abort() -> bool:
        """For wait loops during auto. Abort when switched off with G."""
        poll_g_toggle()
        return not auto_play

    while True:
        frame = capture()
        if frame is None:
            continue

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()

        # G is detected globally so it works even with VRChat in front. Holding does not repeat.
        # While waiting in step / settle, the same detection runs via should_abort.
        poll_g_toggle()

        # Right before key actions or dumps the latest is wanted. Otherwise thinned.
        need_vision = (
            now >= next_vision
            or key in (POLICY_KEY, DUMP_KEY)
            or auto_play
        )
        if need_vision:
            obs = env.observe(frame)
            next_vision = now + 1.0 / VISION_HZ
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

        # p = one move by the policy. During continuous auto with g, drop on every ready.
        # Draw AUTO/LIVE before entering the wait so it shows.
        from_auto = auto_play and key != POLICY_KEY
        if key == POLICY_KEY or (auto_play and obs.ready and not obs.blocked):
            _show(frame, board, obs, aim_x, auto_play, message, message_until, now)
            abort = should_abort if from_auto else None
            if not obs.ready:
                message = "policy: not ready"
            else:
                # Wait until cascades stop before deciding the column.
                obs = wait_playable(env.observe, abort=abort)
                if from_auto and not auto_play:
                    message = "auto=off"
                    frame, obs, board = _refresh(env, frame, obs)
                elif obs.blocked or not obs.ready:
                    message = "policy: not settled"
                    auto_play = False if obs.blocked else auto_play
                    frame, obs, board = _refresh(env, frame, obs)
                else:
                    target = choose_x(obs)
                    aim_x = target
                    result = env.step(target, abort=abort)
                    message = f"auto x={target:.0f} -> {result.info}"
                    aim_x = result.observation.held_x
                    obs = result.observation
                    frame, obs, board = _refresh(env, frame, obs)
                    print(message)
                    if from_auto and not auto_play:
                        message = f"{message} (stop)"
                    elif result.done:
                        auto_play = False
                        message = f"{message} (stop)"
            message_until = now + MESSAGE_SECONDS

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
    hint = "p: policy  g: auto on/off  s: save"
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
