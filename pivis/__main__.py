import logging

import uvicorn

from pivis.config import Settings
from pivis.state import AppState, Queues
from pivis.web.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    settings = Settings()
    queues = Queues()
    app_state = AppState()
    app = create_app(settings, queues, app_state)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
