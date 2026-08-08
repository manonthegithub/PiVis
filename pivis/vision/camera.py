import logging
import threading
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Camera(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_frame(self) -> np.ndarray | None: ...


class PiCamera:
    """picamera2-backed camera. Only works on Raspberry Pi."""

    def __init__(self, resolution: tuple[int, int] = (1280, 720), fps: int = 20, analogue_gain: float = 4.0) -> None:
        self._resolution = resolution
        self._fps = fps
        self._analogue_gain = analogue_gain
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        from picamera2 import Picamera2  # late import — Pi only

        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": self._resolution, "format": "RGB888"},
            controls={
                "FrameRate": self._fps,
                "AeEnable": True,
                "AwbEnable": True,
                "AnalogueGain": self._analogue_gain,
                "AeExposureMode": 0,       # normal AE mode
            },
        )
        self._cam.configure(config)
        self._cam.start()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("PiCamera started at %s @ %dfps", self._resolution, self._fps)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if hasattr(self, "_cam"):
            self._cam.stop()

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _capture_loop(self) -> None:
        while self._running:
            try:
                frame = self._cam.capture_array("main")
                with self._lock:
                    self._frame = frame
            except Exception:
                logger.exception("Camera capture error")


class MockCamera:
    """Solid-color frames for testing off-Pi."""

    def __init__(self, resolution: tuple[int, int] = (1280, 720)) -> None:
        self._resolution = resolution

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_frame(self) -> np.ndarray:
        h, w = self._resolution[1], self._resolution[0]
        return np.zeros((h, w, 3), dtype=np.uint8)


def make_camera(resolution: tuple[int, int], fps: int, analogue_gain: float = 4.0, mock: bool = False) -> Camera:
    if mock:
        return MockCamera(resolution)
    return PiCamera(resolution, fps, analogue_gain)
