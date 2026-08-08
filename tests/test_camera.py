import numpy as np
import pytest
from pivis.vision.camera import MockCamera, make_camera


def test_mock_camera_returns_frame():
    cam = MockCamera(resolution=(640, 480))
    cam.start()
    frame = cam.get_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    cam.stop()


def test_make_camera_mock_flag():
    cam = make_camera(resolution=(1280, 720), fps=20, mock=True)
    cam.start()
    assert cam.get_frame() is not None
    cam.stop()


def test_mock_camera_frame_is_copy():
    cam = MockCamera()
    cam.start()
    f1 = cam.get_frame()
    f2 = cam.get_frame()
    # MockCamera returns new array each call — modifying one shouldn't affect the other
    f1[0, 0, 0] = 255
    assert f2[0, 0, 0] == 0
