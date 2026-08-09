import asyncio
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pivis.vision.camera import MockCamera, PiCamera, make_camera


def test_mock_camera_get_frame():
    cam = MockCamera(resolution=(640, 480))
    cam.start()
    frame = cam.get_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    cam.stop()


def test_mock_camera_get_lores():
    cam = MockCamera()
    cam.start()
    result = cam.get_lores()
    assert result is not None
    frame, ts = result
    assert frame.shape == (320, 320, 3)
    assert isinstance(ts, int)
    _, ts2 = cam.get_lores()
    assert ts2 > ts  # timestamps increment
    cam.stop()


def test_mock_camera_start_h264_pushes_to_queue():
    cam = MockCamera()
    cam.start()
    loop = asyncio.new_event_loop()
    q = asyncio.Queue()
    cam.start_h264(q, loop)
    time.sleep(1.1)  # wait for mock NAL thread to push at 1fps
    cam.stop()
    loop.close()
    assert not q.empty()
    nal, keyframe, pts = q.get_nowait()
    assert isinstance(nal, bytes)
    assert keyframe is True


def test_make_camera_mock_flag():
    cam = make_camera(resolution=(800, 600), fps=20, mock=True)
    cam.start()
    assert cam.get_frame() is not None
    frame, ts = cam.get_lores()
    assert frame.shape == (320, 320, 3)
    cam.stop()


def test_picamera_raises_when_picamera2_missing():
    with patch.dict(sys.modules, {"picamera2": None}):
        cam = PiCamera()
        with pytest.raises((ImportError, ModuleNotFoundError)):
            cam.start()


def test_picamera_get_frame_returns_none_before_start():
    cam = PiCamera()
    assert cam.get_frame() is None


def test_picamera_get_lores_returns_none_before_start():
    cam = PiCamera()
    assert cam.get_lores() is None


def test_picamera_stop_before_start_does_not_crash():
    cam = PiCamera()
    cam.stop()


def test_picamera_start_raises_on_camera_error():
    mock_picamera2 = MagicMock()
    mock_picamera2.Picamera2.side_effect = RuntimeError("No cameras available")
    with patch.dict(sys.modules, {"picamera2": mock_picamera2}):
        cam = PiCamera()
        with pytest.raises(RuntimeError, match="No cameras available"):
            cam.start()
