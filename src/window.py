import ctypes

import cv2

# OpenCV に最大化はないので Win32 でやる。
SW_MAXIMIZE = 3


def maximize_window(title: str) -> None:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)
