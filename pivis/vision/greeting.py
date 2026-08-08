import base64
import logging
import time

import cv2
import numpy as np

from pivis.state import AppState, EventQueue
from pivis.vision.audio import AudioOutput
from pivis.vision.detection import DetectionResult
from pivis.vision.tts import TTSEngine

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a friendly greeter watching a camera feed. "
    "When you see a person, greet them warmly in 1-2 sentences. "
    "Notice something specific and visible about their appearance "
    "(e.g. clothing colour, hair, hat, glasses, bag) and mention it naturally. "
    "Be varied — don't repeat the same greeting. No quotes."
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
        image_b64 = _encode_frame(frame)
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
                        "text": f"{count_str} just walked into view. Greet them, noting something specific about how they look.",
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

        try:
            text = await self._claude.generate_greeting(frame, person_count)
            wav_path = self._tts.synthesize(text)
            await self._audio.play(wav_path, text, self._queue)
        except Exception:
            logger.exception("Greeting pipeline failed — skipping")
