import cv2

from src.capture import capture
from src.calibration import calibrate
from src.vision.board import crop

while True:

    frame = capture()

    if frame is None:
        continue

    key = cv2.waitKey(1) & 0xFF

    # press C key to calibrate
    if key == ord("c"):
        calibrate(frame)

    board = crop(frame)

    if board.size != 0:
        cv2.imshow("Board", board)

    if key == 27:
        break

cv2.destroyAllWindows()