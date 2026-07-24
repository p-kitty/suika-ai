import cv2

from .config import save


def calibrate(frame):

    x, y, w, h = cv2.selectROI(
        "Calibration",
        frame,
        False,
        False
    )

    cv2.destroyWindow("Calibration")

    if w == 0:
        return

    save(
        {
            "board_x": x,
            "board_y": y,
            "board_width": w,
            "board_height": h,
        }
    )