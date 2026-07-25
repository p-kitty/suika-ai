import dxcam

from .config import load

_camera = None


def _get_camera():
    global _camera
    if _camera is None:
        fps = load().get("capture_fps", 15)
        _camera = dxcam.create(output_color="BGR")
        _camera.start(target_fps=fps)
    return _camera


def capture():
    return _get_camera().get_latest_frame()
