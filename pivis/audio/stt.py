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

    def __init__(self, model_name: str = "base", api_key: Optional[str] = None):
        """Initialize Whisper STT.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            api_key: OpenAI API key for cloud API (None = local model)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.local_model = None

        if not api_key:
            try:
                import whisper
                self.local_model = whisper.load_model(model_name)
                logger.info(f"Loaded local Whisper model: {model_name}")
            except ImportError:
                logger.warning("Whisper not installed; falling back to API mode")

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

    async def _transcribe_local(self, audio_bytes: bytes, language: str) -> Dict:
        """Transcribe using local Whisper model."""
        import io
        import numpy as np

        try:
            # Convert PCM bytes to audio array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_data /= 32768.0  # Normalize

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.local_model.transcribe(
                        audio_data,
                        language=language if language != "auto" else None,
                        fp16=False,
                    ),
                ),
                timeout=30.0,
            )

            return {
                "text": result.get("text", "").strip(),
                "confidence": result.get("segments", [{}])[0].get("no_speech_prob", 0.0),
                "language": result.get("language", language),
            }
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
