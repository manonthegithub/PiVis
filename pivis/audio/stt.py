"""Speech-to-text service for audio transcription."""

import asyncio
import logging
from typing import Optional, Dict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class STTBackend(ABC):
    """Abstract base for STT backends."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> Dict:
        """Transcribe audio.

        Returns:
            Dict with keys: text, confidence, language, error (optional)
        """
        pass


class WhisperSTT(STTBackend):
    """Whisper-based STT (local or API)."""

    def __init__(
        self,
        model_name: str = "base",
        api_key: Optional[str] = None,
        beam_size: int = 5,
        cpu_threads: int = 0,
    ):
        """Initialize Whisper STT.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large).
                Must be one of the sizes actually baked into the image (see
                Dockerfile.audio) since HF_HUB_OFFLINE=1 means an uncached
                size fails to load rather than downloading at startup.
            api_key: OpenAI API key for cloud API (None = local model)
            beam_size: whisper's own default is 5; greedy (1) trades
                accuracy for latency. Configurable via BEAM_SIZE.
            cpu_threads: 0 = let CTranslate2 auto-detect, which can
                under-provision relative to an explicit container CPU
                limit. Configurable via CPU_THREADS.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.beam_size = beam_size
        self.local_model = None
        # Whisper's model isn't safe for concurrent .transcribe() calls from
        # multiple threads -- confirmed live: overlapping phrases each ran
        # transcription as their own background task, and two calls landing
        # ~16ms apart produced corrupted-state errors (a zero-element tensor
        # reshape, then a raw nn.Linear object surfacing as an error
        # message) followed by the pod getting OOMKilled shortly after, from
        # the memory footprint of multiple concurrent inferences stacking
        # up. This lock keeps inference calls serialized while still letting
        # the caller's frame-receiving loop stay non-blocking.
        self._infer_lock = asyncio.Lock()

        if not api_key:
            try:
                from faster_whisper import WhisperModel

                # faster-whisper (CTranslate2), not openai-whisper: confirmed
                # live that transcription itself was the main source of
                # latency (~2s/phrase baseline on this ARM node, spiking to
                # 9-13s under node contention). CTranslate2 is typically
                # 3-4x faster for CPU inference at the same model size.
                # int8 quantization trades a little accuracy for further
                # CPU speedup, reasonable on this hardware. cpu_threads=0
                # means "auto-detect", which can under-provision relative
                # to an explicit container CPU limit -- pin it when set.
                self.local_model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cpu_threads,
                )
                logger.info(f"Loaded local faster-whisper model: {model_name}")
            except ImportError:
                logger.warning("faster-whisper not installed; falling back to API mode")

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> Dict:
        """Transcribe audio using Whisper.

        Args:
            audio_bytes: PCM audio data
            language: ISO 639-1 language code

        Returns:
            Transcription result dict
        """
        try:
            if self.local_model:
                return await self._transcribe_local(audio_bytes, language)
            else:
                return await self._transcribe_api(audio_bytes, language)
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": language,
                "error": str(e),
            }

    def _run_faster_whisper(self, audio_data, language: str) -> Dict:
        """Runs in the executor thread: consume faster-whisper's segment
        generator there, not on the event loop thread, and return a plain
        dict -- faster-whisper's Segment objects aren't awaitable/picklable
        concerns, but consuming the generator IS the actual transcription
        work and must stay off the event loop.
        """
        segments, info = self.local_model.transcribe(
            audio_data,
            language=language if language != "auto" else None,
            beam_size=self.beam_size,
            # Without this, whisper always produces *some* text for the
            # given audio duration even when there's no real speech --
            # confirmed repeatedly in testing (pure tone/silence
            # transcribed as "You", "Thanks for watching!", etc., each
            # with a deceptively high confidence). VAD pre-screens for
            # actual speech first; segments genuinely have no speech now
            # come back empty rather than hallucinated, hitting the same
            # empty-segments path already guarded below.
            vad_filter=True,
        )
        segments = list(segments)  # materialize the generator (this is where inference runs)
        text = "".join(s.text for s in segments).strip()
        # faster-whisper's no_speech_prob lives per-segment, same as
        # openai-whisper -- empty segments (no speech detected) previously
        # crashed here with an IndexError on `[][0]`; guard the same way.
        no_speech_prob = segments[0].no_speech_prob if segments else 0.0
        return {
            "text": text,
            "confidence": no_speech_prob,
            "language": info.language or language,
        }

    async def _transcribe_local(self, audio_bytes: bytes, language: str) -> Dict:
        """Transcribe using local Whisper model."""
        import numpy as np

        try:
            # Convert PCM bytes to audio array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_data /= 32768.0  # Normalize

            # Run in thread pool to avoid blocking, but only one inference
            # at a time -- see _infer_lock comment in __init__.
            loop = asyncio.get_event_loop()
            async with self._infer_lock:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self._run_faster_whisper, audio_data, language),
                    timeout=30.0,
                )
            return result
        except asyncio.TimeoutError:
            logger.error("Local Whisper transcription timeout")
            return {
                "text": "",
                "confidence": 0.0,
                "language": language,
                "error": "Transcription timeout",
            }

    async def _transcribe_api(self, audio_bytes: bytes, language: str) -> Dict:
        """Transcribe using OpenAI API."""
        import aiohttp
        import io

        if not self.api_key:
            return {
                "text": "",
                "confidence": 0.0,
                "language": language,
                "error": "No API key provided",
            }

        try:
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field("file", io.BytesIO(audio_bytes), filename="audio.wav")
                form.add_field("model", "whisper-1")
                if language and language != "auto":
                    form.add_field("language", language)

                async with session.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    data=form,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "text": data.get("text", ""),
                            "confidence": 0.95,  # API doesn't return confidence
                            "language": language,
                        }
                    else:
                        error = await resp.text()
                        logger.error(f"Whisper API error: {error}")
                        return {
                            "text": "",
                            "confidence": 0.0,
                            "language": language,
                            "error": f"API error: {resp.status}",
                        }
        except asyncio.TimeoutError:
            return {
                "text": "",
                "confidence": 0.0,
                "language": language,
                "error": "API timeout",
            }
        except Exception as e:
            logger.error(f"STT API call failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": language,
                "error": str(e),
            }


class STTService:
    """Manages STT processing for audio phrases."""

    def __init__(self, backend: Optional[STTBackend] = None):
        """Initialize STT service.

        Args:
            backend: STT backend (defaults to local Whisper)
        """
        self.backend = backend or WhisperSTT()
        self.retry_attempts = 3
        self.retry_delay_ms = 100

    async def transcribe_phrase(
        self, audio_bytes: bytes, language: str = "en"
    ) -> Dict:
        """Transcribe a single phrase with retry logic.

        Args:
            audio_bytes: Segmented audio data
            language: Language code

        Returns:
            Transcription result with error handling
        """
        for attempt in range(self.retry_attempts):
            result = await self.backend.transcribe(audio_bytes, language)

            if not result.get("error"):
                return result

            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(self.retry_delay_ms / 1000.0)
                logger.debug(
                    f"STT retry {attempt + 1}/{self.retry_attempts} for phrase"
                )

        # Final attempt failed
        logger.error(f"STT failed after {self.retry_attempts} attempts")
        return result
