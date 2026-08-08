import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from pivis.config import Settings
from pivis.state import AppState, Queues
from pivis.vision.loop import run_vision_loop
from pivis.web.routes import _attach, router

logger = logging.getLogger(__name__)


def create_app(settings: Settings, queues: Queues, app_state: AppState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(run_vision_loop(settings, queues, app_state))
        logger.info("PiVis started")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info("PiVis stopped")

    app = FastAPI(title="PiVis", lifespan=lifespan)
    _attach(queues, app_state)
    app.include_router(router)
    return app
