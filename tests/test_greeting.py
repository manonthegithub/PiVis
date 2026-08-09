import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from pivis.state import AppState, Queues
from pivis.vision.detection import BoundingBox, DetectionResult
from pivis.vision.greeting import GreetingOrchestrator, _encode_frame


def _result(has_person=True, n_boxes=1):
    boxes = [BoundingBox(0.1, 0.1, 0.5, 0.5, 0.9)] * n_boxes if has_person else []
    return DetectionResult(has_person=has_person, boxes=boxes, confidence=0.9 if has_person else 0.0, timestamp=time.time())


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_orchestrator(state=None, cooldown_s=30):
    claude = AsyncMock()
    claude.generate_greeting = AsyncMock(return_value="Welcome!")

    tts = MagicMock()
    tts.synthesize = MagicMock(return_value=Path("tmp/audio/test.wav"))

    audio = AsyncMock()
    audio.play = AsyncMock()

    state = state or AppState()
    queues = Queues()

    orch = GreetingOrchestrator(
        claude=claude, tts=tts, audio=audio,
        app_state=state, event_queue=queues.events,
        cooldown_s=cooldown_s,
    )
    return orch, claude, tts, audio, state


@pytest.mark.asyncio
async def test_greeting_fires_on_person():
    orch, claude, tts, audio, _ = _make_orchestrator()
    await orch.on_detection(_result(has_person=True), _frame())
    claude.generate_greeting.assert_awaited_once()
    tts.synthesize.assert_called_once_with("Welcome!")
    audio.play.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_greeting_when_no_person():
    orch, claude, tts, audio, _ = _make_orchestrator()
    await orch.on_detection(_result(has_person=False), _frame())
    claude.generate_greeting.assert_not_awaited()
    audio.play.assert_not_awaited()


@pytest.mark.asyncio
async def test_cooldown_blocks_second_greeting():
    state = AppState(last_greeting_at=time.time())
    orch, claude, _, _, _ = _make_orchestrator(state=state, cooldown_s=30)
    await orch.on_detection(_result(has_person=True), _frame())
    claude.generate_greeting.assert_not_awaited()


@pytest.mark.asyncio
async def test_greeting_fires_after_cooldown():
    state = AppState(last_greeting_at=time.time() - 31)
    orch, claude, _, audio, _ = _make_orchestrator(state=state, cooldown_s=30)
    await orch.on_detection(_result(has_person=True), _frame())
    claude.generate_greeting.assert_awaited_once()
    audio.play.assert_awaited_once()


@pytest.mark.asyncio
async def test_claude_failure_falls_back_to_generic_greeting():
    from pivis.vision.greeting import _FALLBACK_GREETINGS
    orch, claude, tts, audio, _ = _make_orchestrator()
    claude.generate_greeting.side_effect = Exception("credit balance too low")
    await orch.on_detection(_result(has_person=True), _frame())  # must not raise
    # Person is still greeted with a fallback line, spoken via TTS + audio.
    spoken = tts.synthesize.call_args[0][0]
    assert spoken in _FALLBACK_GREETINGS
    audio.play.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_failure_does_not_crash():
    orch, claude, tts, audio, _ = _make_orchestrator()
    tts.synthesize.side_effect = Exception("piper missing")
    await orch.on_detection(_result(has_person=True), _frame())  # must not raise
    audio.play.assert_not_awaited()


@pytest.mark.asyncio
async def test_person_count_and_frame_passed_to_claude():
    orch, claude, _, _, _ = _make_orchestrator()
    frame = _frame()
    await orch.on_detection(_result(has_person=True, n_boxes=3), frame)
    args, kwargs = claude.generate_greeting.call_args
    assert args[1] == 3  # person_count
    assert isinstance(args[0], np.ndarray)  # frame passed


def test_encode_frame_returns_base64_string():
    frame = _frame()
    result = _encode_frame(frame)
    assert isinstance(result, str)
    assert len(result) > 0
    import base64
    decoded = base64.b64decode(result)
    assert decoded[:2] == b"\xff\xd8"  # JPEG magic
