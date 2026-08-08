import asyncio
import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from pivis.state import LIGHTING_PRESETS, AppState, Queues

_AUDIO_DIR = Path("tmp/audio")
_STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter()


def _attach(queues: Queues, app_state: AppState) -> None:
    """Inject shared state into route closures."""

    @router.get("/")
    async def index():
        return FileResponse(_STATIC_DIR / "index.html")

    @router.get("/stream")
    async def stream():
        async def _mjpeg():
            while True:
                try:
                    jpeg = await asyncio.wait_for(queues.frames.get(), timeout=1.0)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                except asyncio.TimeoutError:
                    # keepalive empty boundary so browser doesn't disconnect
                    yield b"--frame\r\n\r\n"

        return StreamingResponse(
            _mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @router.get("/events")
    async def events():
        async def _sse():
            app_state.sse_client_count += 1
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queues.events.get(), timeout=30.0)
                        data = json.dumps({"wav_url": event.wav_url, "text": event.text})
                        yield f"data: {data}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                app_state.sse_client_count -= 1

        return StreamingResponse(_sse(), media_type="text/event-stream")

    @router.get("/audio/{name}")
    async def audio(name: str):
        path = _AUDIO_DIR / name
        if not path.exists() or not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, media_type="audio/wav")

    @router.get("/snapshot")
    async def snapshot():
        if app_state.latest_jpeg is None:
            return JSONResponse({"error": "no frame yet"}, status_code=503)
        return Response(content=app_state.latest_jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

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
