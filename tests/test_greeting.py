import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pivis.state import AppState, Queues
from pivis.vision.detection import BoundingBox, DetectionResult
from pivis.vision.greeting import GreetingOrchestrator


def _result(has_person=True, n_boxes=1):
    boxes = [BoundingBox(0.1, 0.1, 0.5, 0.5, 0.9)] * n_boxes if has_person else []
    return DetectionResult(has_person=has_person, boxes=boxes, confidence=0.9 if has_person else 0.0, timestamp=time.time())


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
    orch, claude, tts, audio, state = _make_orchestrator()
    await orch.on_detection(_result(has_person=True))
    claude.generate_greeting.assert_awaited_once_with(1)
    tts.synthesize.assert_called_once_with("Welcome!")
    audio.play.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_greeting_when_no_person():
    orch, claude, tts, audio, _ = _make_orchestrator()
    await orch.on_detection(_result(has_person=False))
    claude.generate_greeting.assert_not_awaited()
    audio.play.assert_not_awaited()


@pytest.mark.asyncio
async def test_cooldown_blocks_second_greeting():
    state = AppState(last_greeting_at=time.time())  # just greeted
    orch, claude, _, _, _ = _make_orchestrator(state=state, cooldown_s=30)
    await orch.on_detection(_result(has_person=True))
    claude.generate_greeting.assert_not_awaited()


@pytest.mark.asyncio
async def test_greeting_fires_after_cooldown():
    state = AppState(last_greeting_at=time.time() - 31)  # cooldown expired
    orch, claude, _, audio, _ = _make_orchestrator(state=state, cooldown_s=30)
    await orch.on_detection(_result(has_person=True))
    claude.generate_greeting.assert_awaited_once()
    audio.play.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_error_does_not_crash():
    orch, claude, _, _, _ = _make_orchestrator()
    claude.generate_greeting.side_effect = Exception("API down")
    await orch.on_detection(_result(has_person=True))  # must not raise


@pytest.mark.asyncio
async def test_person_count_passed_to_claude():
    orch, claude, _, _, _ = _make_orchestrator()
    await orch.on_detection(_result(has_person=True, n_boxes=3))
    claude.generate_greeting.assert_awaited_once_with(3)
