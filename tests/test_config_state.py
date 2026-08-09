import asyncio
import os
import pytest
from pivis.config import Settings
from pivis.state import AppState, AudioEvent, DetectionEvent, Queues


def test_settings_loads_with_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CAMERA_RESOLUTION", raising=False)
    s = Settings(_env_file=None)
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


def test_detection_event():
    ev = DetectionEvent(boxes=[], has_person=False, sensor_timestamp_ns=42)
    assert ev.sensor_timestamp_ns == 42


def test_queues_created():
    q = Queues()
    assert q.events.maxsize == 0
    assert q.controls.maxsize == 0
    assert q.detections.maxsize == 0
    assert q.nal_queue.maxsize == 0
    assert q.fmp4_queue.maxsize == 0
