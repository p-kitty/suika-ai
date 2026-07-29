import ctypes
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.capture import capture
from src.config import load
from src.debug_dump import dump
from src.draw import put_text
from src.env import Env
from src.observe import Observation, clamp_drop_x
from src.policy import choose_x
from src.settle import wait_playable
from src.vision.board import draw_frame_debug
from src.vision.held import DROP_HEIGHT
from src.vision.normalized import inverse_warp_matrix, transform_point, warp_matrix
from src.window import maximize_window

WINDOW_TITLE = "Suika"
MESSAGE_SECONDS = 3.0

DUMP_KEY = ord("s")
DROP_KEY = ord(" ")
POLICY_KEY = ord("p")
LEFT_KEY = ord("a")
RIGHT_KEY = ord("d")
COARSE_LEFT_KEY = ord("j")
COARSE_RIGHT_KEY = ord("l")
CENTER_KEY = ord("c")
QUIT_KEY = 27
# フォーカスに関係なく auto トグルする (VK_G)。
VK_G = 0x47

# 手動で狙うときの左右の刻み (正規化座標, 幅 400)。
# a/d は細かく、j/l は大きく動かす。クリックなら任意の列。
NUDGE = 3.0
COARSE_NUDGE = 15.0
# デバッグ表示用の検出周期。毎フレームフル検出すると重いので間引く。
VISION_HZ = 10.0


@dataclass
class AimClick:
    """クリックで狙い列を受け取る。コールバックから main へ渡す。"""

    x: float | None = None
    corners: np.ndarray | None = field(default=None, repr=False)
    # imshow した画像サイズ。窓が拡大されていてもクリック座標を戻す。
    view_size: tuple[int, int] | None = None


def main() -> None:
    env = Env()
    message = ""
    message_until = 0.0
    next_auto_dump = 0.0
    next_vision = 0.0
    # None のあいだは落下待ちの今の列に合わせる。a/d やクリックで動かしたら固定。
    aim_x: float | None = None
    auto_play = False
    click = AimClick()
    obs = Observation(
        ready=False,
        blocked=False,
        fruits=(),
        held_type=None,
        held_x=None,
        next_type=None,
    )

    maximize_window(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, _on_click, click)
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
            or key
            in (
                LEFT_KEY,
                RIGHT_KEY,
                COARSE_LEFT_KEY,
                COARSE_RIGHT_KEY,
                CENTER_KEY,
                DROP_KEY,
                POLICY_KEY,
                DUMP_KEY,
            )
            or g_pressed
            or click.x is not None
            or auto_play
        )
        if need_vision:
            obs = env.observe(frame)
            next_vision = now + 1.0 / VISION_HZ
        board = env.board

        click.corners = board.corners if board is not None else None

        if obs.ready and obs.held_x is not None:
            if aim_x is None:
                aim_x = obs.held_x
            aim_x = clamp_drop_x(aim_x, obs.held_type)

        if click.x is not None and obs.ready:
            aim_x = clamp_drop_x(click.x, obs.held_type)
            click.x = None

        if key == LEFT_KEY and aim_x is not None:
            aim_x = clamp_drop_x(aim_x - NUDGE, obs.held_type)
        elif key == RIGHT_KEY and aim_x is not None:
            aim_x = clamp_drop_x(aim_x + NUDGE, obs.held_type)
        elif key == COARSE_LEFT_KEY and aim_x is not None:
            aim_x = clamp_drop_x(aim_x - COARSE_NUDGE, obs.held_type)
        elif key == COARSE_RIGHT_KEY and aim_x is not None:
            aim_x = clamp_drop_x(aim_x + COARSE_NUDGE, obs.held_type)
        elif key == CENTER_KEY and obs.held_x is not None:
            aim_x = obs.held_x
        elif g_pressed:
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

        if key == DROP_KEY:
            if not obs.ready or aim_x is None:
                message = "drop: not ready"
            else:
                result = env.step(aim_x)
                message = f"drop x={aim_x:.0f} -> {result.info}"
                aim_x = result.observation.held_x
                obs = result.observation
                # step 中は数秒ブロックするので、ループ先頭の frame は古い。
                frame, obs, board = _refresh(env, frame, obs)
                print(message)
            message_until = now + MESSAGE_SECONDS

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
        hint = f"{mode}  p: policy  g: auto(global)  space: drop  a/d: aim  s: save"
        put_text(output, f"aim x={aim_x:.0f}" if aim_x is not None else "aim —", (8, 128), (0, 255, 255))
        put_text(
            output,
            message if now < message_until else hint,
            (8, output.shape[0] - 12),
            (255, 255, 255),
            scale=0.5,
        )
        click.view_size = (output.shape[1], output.shape[0])
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


def _on_click(event: int, x: int, y: int, _flags: int, click: AimClick) -> None:
    """デバッグ窓のクリック位置を盤面の列に直す。"""
    if event != cv2.EVENT_LBUTTONDOWN or click.corners is None or click.view_size is None:
        return
    # WINDOW_NORMAL + 最大化だとクリック座標は窓サイズ基準になる。
    hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)
    rect = ctypes.wintypes.RECT()
    if hwnd and ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
        win_w = max(1, rect.right - rect.left)
        win_h = max(1, rect.bottom - rect.top)
        img_w, img_h = click.view_size
        x = int(x * img_w / win_w)
        y = int(y * img_h / win_h)
    point = np.array([[[float(x), float(y)]]], dtype=np.float32)
    nx, _ny = cv2.perspectiveTransform(point, warp_matrix(click.corners))[0, 0]
    click.x = float(nx)


def _draw_aim(frame, corners, x: float) -> None:
    matrix = inverse_warp_matrix(corners)
    top = transform_point(matrix, x, -DROP_HEIGHT)
    bottom = transform_point(matrix, x, 40)
    cv2.line(frame, top, bottom, (0, 255, 255), 2)
    cv2.circle(frame, top, 6, (0, 255, 255), -1)


if __name__ == "__main__":
    main()
