import time

import cv2

from src.capture import capture
from src.config import load
from src.debug_dump import dump
from src.draw import put_text
from src.tracker import Tracker
from src.vision.board import draw_frame_debug, localize
from src.window import maximize_window

WINDOW_TITLE = "Suika"
HINT = "s: save raw frame"
MESSAGE_SECONDS = 3.0

DUMP_KEY = ord("s")
QUIT_KEY = 27


def main() -> None:
    tracker = Tracker()
    previous_corners = None
    message = ""
    message_until = 0.0
    next_auto_dump = 0.0

    maximize_window(WINDOW_TITLE)

    while True:
        frame = capture()

        if frame is None:
            continue

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()

        result = localize(frame, previous_corners)
        previous_corners = result.corners

        if result.fruits is None:
            tracker.reset()
        else:
            result.fruits = tracker.update(result.fruits)

        # ゲーム側のウィンドウを操作している間は cv2 にキーが届かないので、
        # 一定間隔で自動保存もできるようにしてある。
        interval = load().get("debug_dump_interval_sec", 0)
        auto = bool(interval) and result.found and now >= next_auto_dump

        if key == DUMP_KEY or auto:
            next_auto_dump = now + max(interval, 1)
            message = dump(frame, result)
            message_until = now + MESSAGE_SECONDS
            print(message)

        output = draw_frame_debug(frame, result)

        put_text(
            output,
            message if now < message_until else HINT,
            (8, output.shape[0] - 12),
            (255, 255, 255),
            scale=0.5,
        )
        cv2.imshow(WINDOW_TITLE, output)

        if key == QUIT_KEY:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
