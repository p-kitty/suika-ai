import cv2

from capture import get_frame
from board import crop_board


while True:
    frame = get_frame()

    if frame is None:
        continue

    board = crop_board(frame)
    cv2.imshow("Board", board)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()