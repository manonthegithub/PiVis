import asyncio
import base64
import logging
import random
import time

import cv2
import numpy as np

from pivis.state import AppState, EventQueue
from pivis.vision.audio import AudioOutput
from pivis.vision.detection import DetectionResult
from pivis.vision.tts import TTSEngine

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a poet watching a camera feed. "
    "When you see a person, compose a short haiku about them — three lines, "
    "roughly 5-7-5 syllables. Notice something specific and visible about their "
    "appearance (clothing colour, hair, hat, glasses, bag) and weave it in. "
    "Be varied and evocative. Output only the haiku, three lines, no quotes or extra text."
)

# Spoken when Claude is unavailable (e.g. no API credits) — a generic haiku so a
# person is still greeted, just without appearance-specific detail.
_FALLBACK_GREETINGS = (
    "A visitor comes,\nquiet steps across the room—\nwelcome, gentle guest.",
    "Someone here at last,\na face bright as morning light—\nhello, traveler.",
    "You appear in view,\nstillness breaks into a smile—\nwelcome, friend, welcome.",
)


def _encode_frame(frame: np.ndarray) -> str:
    """Encode numpy RGB frame as base64 JPEG for Claude vision."""
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode()


class ClaudeClient:
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate_greeting(self, frame: np.ndarray, person_count: int) -> str:
        # JPEG encode is CPU-bound; run it off the event loop so the video stream
        # (sharing this loop) doesn't stall.
        loop = asyncio.get_event_loop()
        image_b64 = await loop.run_in_executor(None, _encode_frame, frame)
        count_str = f"{person_count} person{'s' if person_count > 1 else ''}"
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=80,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                    },
                    {
                        "type": "text",
                        "text": f"{count_str} just came into view. Write a haiku about them, noting something specific about how they look.",
                    },
                ],
            }],
            timeout=10.0,
        )
        return message.content[0].text.strip()


class GreetingOrchestrator:
    def __init__(
        self,
        claude: ClaudeClient,
        tts: TTSEngine,
        audio: AudioOutput,
        app_state: AppState,
        event_queue: EventQueue,
        cooldown_s: int = 30,
    ) -> None:
        self._claude = claude
        self._tts = tts
        self._audio = audio
        self._state = app_state
        self._queue = event_queue
        self._cooldown_s = cooldown_s

    async def on_detection(self, result: DetectionResult, frame: np.ndarray) -> None:
        if not result.has_person:
            return
        if time.time() - self._state.last_greeting_at < self._cooldown_s:
            return

        self._state.last_greeting_at = time.time()
        person_count = len(result.boxes)

        # Claude adds appearance-specific detail but is optional — fall back to a
        # generic line so the person is still greeted if the API is unavailable.
        try:
            text = await self._claude.generate_greeting(frame, person_count)
        except Exception as exc:
            text = random.choice(_FALLBACK_GREETINGS)
            logger.warning("Claude greeting unavailable, using fallback: %s", exc)

        try:
            # Piper runs a blocking subprocess; offload it so the video stream
            # (same event loop) keeps flowing while speech is synthesized.
            loop = asyncio.get_event_loop()
            wav_path = await loop.run_in_executor(None, self._tts.synthesize, text)
            await self._audio.play(wav_path, text, self._queue)
        except Exception:
            logger.exception("TTS/audio failed — no greeting played")
