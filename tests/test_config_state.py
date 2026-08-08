import asyncio
import os
import pytest
from pivis.config import Settings
from pivis.state import AppState, AudioEvent, Queues


def test_settings_loads_with_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    s = Settings()
    assert s.anthropic_api_key == "test-key"
    assert s.stream_fps == 20
    assert s.greeting_cooldown_s == 30
    assert s.audio_output == "both"


def test_settings_missing_api_key_raises():
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with pytest.MonkeyPatch().context() as m:
        m.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(Exception):
            Settings(_env_file=None)


def test_app_state_defaults():
    state = AppState()
    assert state.has_person is False
    assert state.last_greeting_at == 0.0
    assert state.sse_client_count == 0


def test_audio_event():
    ev = AudioEvent(wav_url="/audio/test.wav", text="Hello!")
    assert ev.wav_url == "/audio/test.wav"
    assert ev.text == "Hello!"


def test_queues_created():
    q = Queues()
    assert q.frames.maxsize == 2
    assert q.events.maxsize == 0  # unbounded


def test_frame_queue_drops_old_when_full():
    async def _run():
        q = Queues()
        await q.frames.put(b"frame1")
        await q.frames.put(b"frame2")
        assert q.frames.full()

    asyncio.run(_run())
