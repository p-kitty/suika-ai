import dxcam

camera = dxcam.create(output_color="BGR")
camera.start(target_fps=60)


def get_frame():
    return camera.get_latest_frame()