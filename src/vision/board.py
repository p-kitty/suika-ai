from ..config import load

def crop(frame):
    cfg = load()

    x = cfg["board_x"]
    y = cfg["board_y"]
    w = cfg["board_width"]
    h = cfg["board_height"]

    return frame[y:y+h, x:x+w]