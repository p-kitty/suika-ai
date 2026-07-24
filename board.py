from config import *

def crop_board(frame):
    return frame[
        BOARD_Y:BOARD_Y + BOARD_HEIGHT,
        BOARD_X:BOARD_X + BOARD_WIDTH
    ]