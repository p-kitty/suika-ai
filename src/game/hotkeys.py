import ctypes


class EdgeKey:
    """グローバルキーの立ち上がり検出 (GetAsyncKeyState)。"""

    def __init__(self, vk: int) -> None:
        self.vk = vk
        self._was_down = False

    def poll(self) -> bool:
        """押された瞬間だけ True。"""
        down = bool(ctypes.windll.user32.GetAsyncKeyState(self.vk) & 0x8000)
        pressed = down and not self._was_down
        self._was_down = down
        return pressed
