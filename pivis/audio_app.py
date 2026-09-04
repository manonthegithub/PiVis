"""Standalone runner for the audio module: mic -> STT -> text back to browser.

Deliberately separate from the main PiVis app (pivis.web.app). The main app's
lifespan hard-launches the camera/vision loop, which needs real Pi camera
hardware; this app only exercises the browser audio capture -> WebSocket ->
speech-to-text path, with no camera, no TTS/greeting, and no LLM step for now
(transcribed text is sent straight back to the browser).
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pivis.audio.processor import AudioProcessor
from pivis.audio.stream_handler import AudioStreamHandler
from pivis.audio.stt import STTService, WhisperSTT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "web" / "static"
_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")

app = FastAPI(title="PiVis Audio Module")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Loaded once at import time (module download is baked into the image at
# build time so pod startup doesn't need network access).
stt_service = STTService(backend=WhisperSTT(model_name=_WHISPER_MODEL))

# Per-stream framing/silence-detection state, keyed by stream_id.
_processors: dict[str, AudioProcessor] = {}
# In-flight background transcription tasks, keyed by stream_id, so a
# disconnect can cancel them instead of leaking or writing to a dead socket.
_pending_tasks: dict[str, set] = {}


async def _transcribe_and_reply(
    stream_id: str, websocket, audio_data: bytes, duration_ms: int
) -> None:
    """Run STT and deliver the result, off the connection's receive loop.

    Whisper transcription is CPU-bound and can take tens of seconds on this
    hardware (confirmed live: ~40s pinned at the pod's 1-core CPU limit for
    a single real phrase). Awaiting it inline in _on_frame used to block
    that connection from reading any further audio frames for the whole
    duration -- nothing crashed, but the stream looked completely hung with
    no feedback. Running it as a background task keeps frames flowing.
    """
    try:
        transcription = await stt_service.transcribe_phrase(audio_data)
        if transcription.get("error"):
            await websocket.send_text(
                json.dumps(
                    {"type": "error", "message": transcription["error"], "duration_ms": duration_ms}
                )
            )
        else:
            await websocket.send_text(
                json.dumps({"type": "transcription", "duration_ms": duration_ms, **transcription})
            )
    except Exception as e:
        # Most likely the websocket closed while transcription was still
        # running -- nothing to deliver to, not worth an error log.
        logger.debug(f"Stream {stream_id}: couldn't deliver transcription - {e}")


async def _on_frame(stream_id: str, frame: dict) -> None:
    processor = _processors.setdefault(
        stream_id, AudioProcessor(sample_rate=frame.get("sample_rate", 16000))
    )
    result = processor.process_frame(frame["audio_data"], frame["frame_id"])
    if not result:
        return

    stream_state = stream_handler.active_streams.get(stream_id)
    if not stream_state:
        return
    websocket = stream_state["websocket"]

    for sub_frame in result["frames"]:
        if sub_frame["type"] != "phrase_complete":
            continue
        # Tell the client transcription has started -- without this, a
        # 20-40s wait with zero messages is indistinguishable from a hang.
        # duration_ms exposes exactly where/how long each silence-detected
        # segment is, so the sentence-splitting behavior is visible instead
        # of a black box.
        duration_ms = sub_frame.get("duration_ms", 0)
        await websocket.send_text(
            json.dumps({"type": "processing", "duration_ms": duration_ms})
        )
        task = asyncio.create_task(
            _transcribe_and_reply(stream_id, websocket, sub_frame["audio_data"], duration_ms)
        )
        _pending_tasks.setdefault(stream_id, set()).add(task)
        task.add_done_callback(
            lambda t, sid=stream_id: _pending_tasks.get(sid, set()).discard(t)
        )


stream_handler = AudioStreamHandler(frame_callback=_on_frame)


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "audio_test.html")


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "whisper_model": _WHISPER_MODEL}


@app.websocket("/ws/audio/{stream_id}")
async def ws_audio(websocket: WebSocket, stream_id: str):
    try:
        await stream_handler.handle_connection(websocket, stream_id)
    finally:
        _processors.pop(stream_id, None)
        for task in _pending_tasks.pop(stream_id, set()):
            task.cancel()
