import asyncio
import logging
import threading
import time
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

_PTS_HZ = 90_000  # MPEG standard time_base for H.264


class Camera(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_frame(self) -> np.ndarray | None: ...
    def get_lores(self) -> tuple[np.ndarray, int] | None: ...  # (frame, sensor_timestamp_ns)
    def set_controls(self, controls: dict) -> None: ...
    def start_h264(self, nal_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None: ...
    def stop_h264(self) -> None: ...


def _make_nal_output(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Build an Output-subclass instance that routes encoder NALs to an asyncio queue.

    Constructed lazily so picamera2 is only imported on-device (not in tests/off-Pi).
    """
    from picamera2.outputs import Output

    class NalQueueOutput(Output):
        """Routes picamera2 H264Encoder output to an asyncio queue."""

        def __init__(self) -> None:
            super().__init__()
            self._queue = queue
            self._loop = loop

        def outputframe(
            self,
            data: bytes,
            keyframe: bool = True,
            timestamp: int | None = None,
            packet=None,
            audio: bool = False,
        ) -> None:
            # picamera2's Encoder calls this positionally as (frame, keyframe, timestamp, packet, audio).
            if audio or not data:
                return
            # timestamp is PTS from the encoder in microseconds; convert to 90kHz ticks.
            pts_ticks = (timestamp or 0) * _PTS_HZ // 1_000_000
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (data, keyframe, pts_ticks))

    return NalQueueOutput()


class PiCamera:
    """picamera2-backed camera with dual-stream (main→H264, lores→YOLO)."""

    def __init__(self, resolution: tuple[int, int] = (800, 600), fps: int = 20, analogue_gain: float = 4.0) -> None:
        self._resolution = resolution
        self._fps = fps
        self._analogue_gain = analogue_gain
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._lores_frame: np.ndarray | None = None
        self._lores_ts_ns: int = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._encoder = None

    def start(self) -> None:
        from picamera2 import Picamera2

        self._cam = Picamera2()
        lores_size = (320, 320)
        config = self._cam.create_video_configuration(
            main={"size": self._resolution, "format": "YUV420"},
            lores={"size": lores_size, "format": "RGB888"},
            controls={
                "FrameRate": self._fps,
                "AeEnable": True,
                "AwbEnable": True,
                "AnalogueGain": self._analogue_gain,
                "AeExposureMode": 0,
            },
        )
        self._cam.configure(config)
        self._cam.start()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("PiCamera started at %s + lores %s @ %dfps", self._resolution, lores_size, self._fps)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.stop_h264()
        if hasattr(self, "_cam"):
            self._cam.stop()

    def get_frame(self) -> np.ndarray | None:
        """Return latest main frame (RGB) for Claude vision API."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_lores(self) -> tuple[np.ndarray, int] | None:
        """Return (lores_rgb_frame, sensor_timestamp_ns) for YOLO."""
        with self._lock:
            if self._lores_frame is None:
                return None
            return self._lores_frame.copy(), self._lores_ts_ns

    def set_controls(self, controls: dict) -> None:
        if hasattr(self, "_cam"):
            self._cam.set_controls(controls)
            logger.info("Camera controls updated: %s", controls)

    def start_h264(self, nal_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        from picamera2.encoders import H264Encoder

        self._nal_output = _make_nal_output(nal_queue, loop)
        # profile="main" matches the browser MediaSource codec string (avc1.4D401E);
        # repeat=True re-emits SPS/PPS with every keyframe so late-joining clients can decode.
        self._encoder = H264Encoder(bitrate=1_000_000, repeat=True, profile="main")
        self._encoder.output = self._nal_output
        self._cam.start_encoder(self._encoder)
        logger.info("H264Encoder started")

    def stop_h264(self) -> None:
        if self._encoder is not None and hasattr(self, "_cam"):
            try:
                self._cam.stop_encoder()
            except Exception:
                pass
            self._encoder = None

    def _capture_loop(self) -> None:
        import cv2
        while self._running:
            try:
                request = self._cam.capture_request()
                lores = request.make_array("lores")
                ts_ns = request.get_metadata().get("SensorTimestamp", 0)
                # main is YUV420; convert to RGB for Claude greeting
                main_yuv = request.make_array("main")
                request.release()
                main_rgb = cv2.cvtColor(main_yuv, cv2.COLOR_YUV420p2RGB)
                with self._lock:
                    self._lores_frame = lores
                    self._lores_ts_ns = ts_ns
                    self._frame = main_rgb
            except Exception:
                logger.exception("Camera capture error")


class MockCamera:
    """Synthetic frames for testing off-Pi."""

    def __init__(self, resolution: tuple[int, int] = (800, 600)) -> None:
        self._resolution = resolution
        self._ts: int = 0
        self._nal_queue: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._nal_thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.stop_h264()

    def get_frame(self) -> np.ndarray:
        h, w = self._resolution[1], self._resolution[0]
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_lores(self) -> tuple[np.ndarray, int]:
        self._ts += 50_000_000  # 50ms in nanoseconds (20fps)
        return np.zeros((320, 320, 3), dtype=np.uint8), self._ts

    def set_controls(self, controls: dict) -> None:
        pass

    def start_h264(self, nal_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self._nal_queue = nal_queue
        self._loop = loop
        self._nal_thread = threading.Thread(target=self._push_nals, daemon=True)
        self._nal_thread.start()

    def stop_h264(self) -> None:
        self._running = False
        if self._nal_thread:
            self._nal_thread.join(timeout=1)
            self._nal_thread = None

    def _push_nals(self) -> None:
        pts = 0
        while self._running and self._nal_queue:
            nal = b"\x00\x00\x00\x01\x65" + b"\x00" * 64  # minimal Annex-B IDR
            # put_nowait is safe from a thread in CPython (GIL protects the deque)
            self._nal_queue.put_nowait((nal, True, pts))
            pts += _PTS_HZ  # 1fps synthetic
            time.sleep(1.0)


def make_camera(resolution: tuple[int, int], fps: int, analogue_gain: float = 4.0, mock: bool = False) -> Camera:
    if mock:
        return MockCamera(resolution)
    return PiCamera(resolution, fps, analogue_gain)
