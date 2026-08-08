import logging
import time

from pivis.state import AppState, EventQueue
from pivis.vision.audio import AudioOutput
from pivis.vision.detection import DetectionResult
from pivis.vision.tts import TTSEngine

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a friendly greeter. When a person is detected by a camera, "
    "generate a short, warm, welcoming greeting (1-2 sentences). "
    "Be natural and varied — don't repeat the same greeting. No quotes."
)


class ClaudeClient:
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate_greeting(self, person_count: int) -> str:
        user_msg = (
            f"{person_count} person{'s' if person_count > 1 else ''} "
            "just walked into view. Generate a greeting."
        )
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=64,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
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

    async def on_detection(self, result: DetectionResult) -> None:
        if not result.has_person:
            return
        if time.time() - self._state.last_greeting_at < self._cooldown_s:
            return

        self._state.last_greeting_at = time.time()
        person_count = len(result.boxes)

        try:
            text = await self._claude.generate_greeting(person_count)
            wav_path = self._tts.synthesize(text)
            await self._audio.play(wav_path, text, self._queue)
        except Exception:
            logger.exception("Greeting pipeline failed — skipping")
