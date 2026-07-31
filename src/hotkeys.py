import ctypes


class EdgeKey:
    """Rising-edge detection of global keys (GetAsyncKeyState)."""

    def __init__(self, vk: int) -> None:
        self.vk = vk
        self._was_down = False

    def poll(self) -> bool:
        """True only at the moment of the press."""
        down = bool(ctypes.windll.user32.GetAsyncKeyState(self.vk) & 0x8000)
        pressed = down and not self._was_down
        self._was_down = down
        return pressed
