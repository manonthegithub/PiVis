import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from pivis.state import LIGHTING_PRESETS, AppState, Queues

_AUDIO_DIR = Path("tmp/audio")
_STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter()


def _attach(queues: Queues, app_state: AppState) -> None:
    """Inject shared state into route closures."""

    @router.get("/")
    async def index():
        return FileResponse(_STATIC_DIR / "index.html")

    @router.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                chunk = await queues.fmp4_queue.get()
                await websocket.send_bytes(chunk)
        except (WebSocketDisconnect, Exception):
            pass

    @router.get("/events")
    async def events():
        async def _sse():
            app_state.sse_client_count += 1
            stream_start_sent = False
            try:
                while True:
                    # Wait for whichever queue has data first
                    audio_task = asyncio.ensure_future(queues.events.get())
                    det_task = asyncio.ensure_future(queues.detections.get())
                    done, pending = await asyncio.wait(
                        {audio_task, det_task},
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=30.0,
                    )
                    for t in pending:
                        t.cancel()

                    if not done:
                        yield ": keepalive\n\n"
                        continue

                    for t in done:
                        item = t.result()
                        from pivis.state import AudioEvent, DetectionEvent
                        if isinstance(item, AudioEvent):
                            data = json.dumps({"wav_url": item.wav_url, "text": item.text})
                            yield f"event: audio\ndata: {data}\n\n"
                        elif isinstance(item, DetectionEvent):
                            if not stream_start_sent:
                                start_data = json.dumps({
                                    "sensor_timestamp_ns": item.sensor_timestamp_ns,
                                    "pts_seconds": 0.0,
                                })
                                yield f"event: stream_start\ndata: {start_data}\n\n"
                                stream_start_sent = True
                            det_data = json.dumps({
                                "boxes": item.boxes,
                                "has_person": item.has_person,
                                "sensor_timestamp_ns": item.sensor_timestamp_ns,
                            })
                            yield f"event: detection\ndata: {det_data}\n\n"
            finally:
                app_state.sse_client_count -= 1

        return StreamingResponse(_sse(), media_type="text/event-stream")

    @router.get("/audio/{name}")
    async def audio(name: str):
        path = _AUDIO_DIR / name
        if not path.exists() or not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, media_type="audio/wav")

    @router.get("/status")
    async def status():
        return {
            "has_person": app_state.has_person,
            "last_greeting_at": app_state.last_greeting_at,
            "sse_client_count": app_state.sse_client_count,
        }

    @router.post("/settings/lighting/{preset}")
    async def set_lighting(preset: str):
        if preset not in LIGHTING_PRESETS:
            return JSONResponse({"error": f"unknown preset, use: {list(LIGHTING_PRESETS)}"}, status_code=400)
        await queues.controls.put(LIGHTING_PRESETS[preset])
        return {"preset": preset, "controls": LIGHTING_PRESETS[preset]}
