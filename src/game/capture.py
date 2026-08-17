import dxcam

CAPTURE_FPS = 15

_camera = None


def _get_camera():
    global _camera
    if _camera is None:
        _camera = dxcam.create(output_color="BGR")
        _camera.start(target_fps=CAPTURE_FPS)
    return _camera


def capture():
    return _get_camera().get_latest_frame()
