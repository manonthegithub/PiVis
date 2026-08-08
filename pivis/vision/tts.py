import logging
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_AUDIO_DIR = Path("tmp/audio")


class TTSEngine:
    def __init__(self, piper_binary: str, voice_path: Path) -> None:
        self._piper = piper_binary
        self._voice = voice_path
        _AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str) -> Path:
        if not self._voice.exists():
            raise FileNotFoundError(f"Piper voice model not found: {self._voice}")

        out_path = _AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
        try:
            result = subprocess.run(
                [self._piper, "--model", str(self._voice), "--output_file", str(out_path)],
                input=text.encode(),
                capture_output=True,
                timeout=15,
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Piper binary not found: {self._piper!r}. Install piper-tts.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Piper TTS timed out after 15s")

        if result.returncode != 0:
            raise RuntimeError(f"Piper failed (exit {result.returncode}): {result.stderr.decode()}")

        return out_path

    @staticmethod
    def delete(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete audio file: %s", path)
