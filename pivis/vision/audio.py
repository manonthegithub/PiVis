import asyncio
import logging
from pathlib import Path
from typing import Protocol

from pivis.state import AudioEvent, EventQueue
from pivis.vision.tts import TTSEngine

logger = logging.getLogger(__name__)

_BROWSER_FETCH_RETENTION_S = 30  # keep the WAV around long enough for the browser to GET it


class AudioOutput(Protocol):
    async def play(self, wav_path: Path, text: str, event_queue: EventQueue) -> None: ...


async def _delete_after(wav_path: Path, delay_s: float) -> None:
    await asyncio.sleep(delay_s)
    TTSEngine.delete(wav_path)


class BrowserAudioOutput:
    """Pushes AudioEvent to SSE queue; browser fetches and plays the WAV.

    The file must outlive the SSE event so the browser's GET /audio/... succeeds,
    so deletion is deferred rather than immediate.
    """

    async def play(self, wav_path: Path, text: str, event_queue: EventQueue) -> None:
        await event_queue.put(AudioEvent(wav_url=f"/audio/{wav_path.name}", text=text))
        asyncio.create_task(_delete_after(wav_path, _BROWSER_FETCH_RETENTION_S))


class LocalAudioOutput:
    """Plays WAV directly via aplay on Pi speakers."""

    def __init__(self, device: str = "default", cleanup: bool = True) -> None:
        self._device = device
        self._cleanup = cleanup  # False when a browser output owns the file's lifetime

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
            if self._cleanup:
                TTSEngine.delete(wav_path)


class BothAudioOutput:
    """Plays on speaker AND pushes to browser simultaneously."""

    def __init__(self, device: str = "default") -> None:
        self._browser = BrowserAudioOutput()
        # Browser output owns deletion (after a fetch window); local must not delete
        # the file out from under the browser's pending GET.
        self._local = LocalAudioOutput(device, cleanup=False)

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
