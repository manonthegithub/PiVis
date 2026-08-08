import asyncio
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from pivis.config import Settings
from pivis.state import AppState, Queues
from pivis.vision.audio import make_audio_output
from pivis.vision.camera import Camera, make_camera
from pivis.vision.detection import BoundingBox, DetectionEngine
from pivis.vision.greeting import ClaudeClient, GreetingOrchestrator
from pivis.vision.tts import TTSEngine

logger = logging.getLogger(__name__)


def encode_jpeg(frame: np.ndarray, boxes: list[BoundingBox]) -> bytes:
    """Draw bounding boxes on a copy and encode to JPEG bytes."""
    h, w = frame.shape[:2]
    out = frame.copy()
    for box in boxes:
        x1, y1 = int(box.x1 * w), int(box.y1 * h)
        x2, y2 = int(box.x2 * w), int(box.y2 * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


async def run_vision_loop(settings: Settings, queues: Queues, app_state: AppState) -> None:
    camera = make_camera((settings.camera_width, settings.camera_height), settings.stream_fps)
    engine = DetectionEngine(confidence_threshold=settings.detection_confidence)
    tts = TTSEngine(settings.piper_binary, settings.tts_voice_path)
    audio = make_audio_output(settings.audio_output, settings.audio_device)
    claude = ClaudeClient(settings.anthropic_api_key, settings.claude_model)
    orchestrator = GreetingOrchestrator(
        claude=claude, tts=tts, audio=audio,
        app_state=app_state, event_queue=queues.events,
        cooldown_s=settings.greeting_cooldown_s,
    )

    engine.load(settings.yolo_model_path)
    camera.start()
    logger.info("Vision loop started")

    interval_s = settings.detection_interval_ms / 1000
    last_detect_t = 0.0

    try:
        while True:
            frame = await asyncio.get_event_loop().run_in_executor(None, camera.get_frame)
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            now = time.monotonic()
            boxes = []

            if now - last_detect_t >= interval_s:
                last_detect_t = now
                try:
                    result = await asyncio.get_event_loop().run_in_executor(None, engine.detect, frame)
                    boxes = result.boxes
                    app_state.has_person = result.has_person
                    asyncio.create_task(orchestrator.on_detection(result, frame))
                except Exception:
                    logger.exception("Detection error")

            jpeg = encode_jpeg(frame, boxes)
            try:
                queues.frames.put_nowait(jpeg)
            except asyncio.QueueFull:
                try:
                    queues.frames.get_nowait()  # drop oldest
                    queues.frames.put_nowait(jpeg)
                except asyncio.QueueEmpty:
                    pass

            await asyncio.sleep(1 / settings.stream_fps)

    finally:
        camera.stop()
        logger.info("Vision loop stopped")
