import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from pivis.config import Settings
from pivis.state import AppState, Queues
from pivis.vision.fmp4 import run_fmp4_loop
from pivis.vision.loop import run_vision_loop
from pivis.vision.webrtc import NalHub, close_all
from pivis.web.eventhub import EventHub
from pivis.web.routes import _attach, router

logger = logging.getLogger(__name__)


def create_app(settings: Settings, queues: Queues, app_state: AppState) -> FastAPI:
    webrtc = settings.stream_mode == "webrtc"
    hub = NalHub(queues.nal_queue) if webrtc else None
    pcs: set = set()
    event_hub = EventHub(queues.events, queues.detections)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        event_hub.start()
        vision_task = asyncio.create_task(run_vision_loop(settings, queues, app_state))
        if webrtc:
            hub.start()
            backend_task = None
        else:
            backend_task = asyncio.create_task(
                run_fmp4_loop(queues.nal_queue, queues.fmp4_queue, fps=settings.stream_fps, queues=queues)
            )
        logger.info("PiVis started (stream_mode=%s)", settings.stream_mode)
        try:
            yield
        finally:
            vision_task.cancel()
            tasks = [vision_task] + ([backend_task] if backend_task else [])
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            if webrtc:
                await close_all(pcs)
                await hub.stop()
            await event_hub.stop()
            logger.info("PiVis stopped")

    app = FastAPI(title="PiVis", lifespan=lifespan)
    _attach(queues, app_state, settings=settings, hub=hub, pcs=pcs, event_hub=event_hub)
    app.include_router(router)
    return app
