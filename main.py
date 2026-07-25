import cv2

from src.capture import capture
from src.vision.board import draw_frame_debug, localize

while True:
    frame = capture()

    if frame is None:
        continue

    key = cv2.waitKey(1) & 0xFF

    result = localize(frame)
    cv2.imshow("Frame", draw_frame_debug(frame, result))

    if key == 27:
        break

cv2.destroyAllWindows()
