import asyncio
import logging
from pathlib import Path
from typing import Protocol

from pivis.state import AudioEvent, EventQueue
from pivis.vision.tts import TTSEngine

logger = logging.getLogger(__name__)


class AudioOutput(Protocol):
    async def play(self, wav_path: Path, text: str, event_queue: EventQueue) -> None: ...


class BrowserAudioOutput:
    """Pushes AudioEvent to SSE queue; browser plays the WAV."""

    async def play(self, wav_path: Path, text: str, event_queue: EventQueue) -> None:
        await event_queue.put(AudioEvent(wav_url=f"/audio/{wav_path.name}", text=text))


class LocalAudioOutput:
    """Plays WAV directly via aplay on Pi speakers."""

    def __init__(self, device: str = "default") -> None:
        self._device = device

    async def play(self, wav_path: Path, text: str, event_queue: EventQueue) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "aplay", "-D", self._device, str(wav_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning("aplay error (exit %d): %s", proc.returncode, stderr.decode())
        except FileNotFoundError:
            logger.error("aplay not found — install alsa-utils on the Pi")
        finally:
            TTSEngine.delete(wav_path)


class BothAudioOutput:
    """Plays on speaker AND pushes to browser simultaneously."""

    def __init__(self, device: str = "default") -> None:
        self._browser = BrowserAudioOutput()
        self._local = LocalAudioOutput(device)

    async def play(self, wav_path: Path, text: str, event_queue: EventQueue) -> None:
        await asyncio.gather(
            self._browser.play(wav_path, text, event_queue),
            self._local.play(wav_path, text, event_queue),
        )


def make_audio_output(mode: str, device: str = "default") -> AudioOutput:
    match mode:
        case "browser":
            return BrowserAudioOutput()
        case "local":
            return LocalAudioOutput(device)
        case "both":
            return BothAudioOutput(device)
        case _:
            raise ValueError(f"Unknown audio_output mode: {mode!r}. Use 'browser', 'local', or 'both'.")
