import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pivis.state import AppState, AudioEvent, Queues
from pivis.web.routes import APIRouter, _attach

# Build a minimal app without triggering vision loop
from fastapi import FastAPI


def _make_test_app(queues: Queues, app_state: AppState) -> FastAPI:
    app = FastAPI()
    _attach(queues, app_state)
    from pivis.web.routes import router
    app.include_router(router)
    return app


@pytest.fixture
def client_ctx():
    queues = Queues()
    state = AppState(has_person=True, last_greeting_at=1234.0, sse_client_count=0)
    app = _make_test_app(queues, state)
    return TestClient(app, raise_server_exceptions=True), queues, state


def test_status_endpoint(client_ctx):
    client, _, state = client_ctx
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_person"] is True
    assert data["last_greeting_at"] == 1234.0


def test_audio_404_missing(client_ctx):
    client, _, _ = client_ctx
    resp = client.get("/audio/nonexistent.wav")
    assert resp.status_code == 404


def test_audio_served(client_ctx, tmp_path):
    client, _, _ = client_ctx
    wav = Path("tmp/audio/test.wav")
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"RIFF")
    try:
        resp = client.get("/audio/test.wav")
        assert resp.status_code == 200
        assert resp.content == b"RIFF"
    finally:
        wav.unlink(missing_ok=True)


def test_index_serves_html(client_ctx, tmp_path):
    client, _, _ = client_ctx
    fake_html = b"<html></html>"
    static = Path(__file__).parent.parent / "pivis/web/static/index.html"
    static.parent.mkdir(parents=True, exist_ok=True)
    static.write_bytes(fake_html)
    resp = client.get("/")
    assert resp.status_code == 200
