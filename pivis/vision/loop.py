import asyncio
import logging
import time

from pivis.config import Settings
from pivis.state import AppState, DetectionEvent, Queues
from pivis.vision.audio import make_audio_output
from pivis.vision.camera import make_camera
from pivis.vision.detection import DetectionEngine
from pivis.vision.greeting import ClaudeClient, GreetingOrchestrator
from pivis.vision.tts import TTSEngine

logger = logging.getLogger(__name__)


async def run_vision_loop(settings: Settings, queues: Queues, app_state: AppState) -> None:
    camera = make_camera(
        (settings.camera_width, settings.camera_height),
        settings.stream_fps,
        settings.camera_analogue_gain,
    )
    engine = DetectionEngine(
        confidence_threshold=settings.detection_confidence,
        input_size=settings.detection_input_size,
    )
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
    loop = asyncio.get_event_loop()
    try:
        camera.start_h264(queues.nal_queue, loop)
    except Exception:
        logger.exception("Failed to start H264 encoder; video stream disabled")
    logger.info("Vision loop started")

    interval_s = settings.detection_interval_ms / 1000
    last_detect_t = 0.0

    try:
        while True:
            # Apply any pending camera control changes
            while not queues.controls.empty():
                try:
                    camera.set_controls(queues.controls.get_nowait())
                except Exception:
                    logger.exception("Failed to apply camera controls")

            lores = await asyncio.get_event_loop().run_in_executor(None, camera.get_lores)
            if lores is None:
                await asyncio.sleep(0.05)
                continue

            lores_frame, sensor_ts_ns = lores
            now = time.monotonic()

            if now - last_detect_t >= interval_s:
                last_detect_t = now
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, engine.detect, lores_frame, sensor_ts_ns
                    )
                    app_state.has_person = result.has_person
                    boxes = [
                        {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "confidence": b.confidence}
                        for b in result.boxes
                    ]
                    await queues.detections.put(DetectionEvent(
                        boxes=boxes,
                        has_person=result.has_person,
                        sensor_timestamp_ns=sensor_ts_ns,
                    ))
                    if result.has_person:
                        # get_frame() returns full-res RGB for Claude vision API
                        frame = await asyncio.get_event_loop().run_in_executor(None, camera.get_frame)
                        if frame is not None:
                            asyncio.create_task(orchestrator.on_detection(result, frame))
                except Exception:
                    logger.exception("Detection error")

            await asyncio.sleep(1 / settings.stream_fps)

    finally:
        camera.stop_h264()
        camera.stop()
        logger.info("Vision loop stopped")
