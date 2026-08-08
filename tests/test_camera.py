import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pivis.vision.camera import MockCamera, PiCamera, make_camera


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
    f1[0, 0, 0] = 255
    assert f2[0, 0, 0] == 0


def test_picamera_raises_when_picamera2_missing():
    """PiCamera.start() must raise ImportError if picamera2/libcamera not installed."""
    with patch.dict(sys.modules, {"picamera2": None}):
        cam = PiCamera()
        with pytest.raises((ImportError, ModuleNotFoundError)):
            cam.start()


def test_picamera_get_frame_returns_none_before_start():
    cam = PiCamera()
    assert cam.get_frame() is None


def test_picamera_stop_before_start_does_not_crash():
    cam = PiCamera()
    cam.stop()  # must not raise


def test_picamera_start_raises_on_camera_error():
    """If Picamera2() itself raises (no camera hardware), start() should propagate."""
    mock_picamera2 = MagicMock()
    mock_picamera2.Picamera2.side_effect = RuntimeError("No cameras available")
    with patch.dict(sys.modules, {"picamera2": mock_picamera2}):
        cam = PiCamera()
        with pytest.raises(RuntimeError, match="No cameras available"):
            cam.start()
