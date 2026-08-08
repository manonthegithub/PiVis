import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pivis.vision.tts import TTSEngine


@pytest.fixture
def engine(tmp_path):
    voice = tmp_path / "voice.onnx"
    voice.touch()
    with patch("pivis.vision.tts._AUDIO_DIR", tmp_path):
        yield TTSEngine(piper_binary="piper", voice_path=voice), tmp_path


def test_synthesize_success(engine):
    tts, tmp_path = engine
    fake_wav = tmp_path / "output.wav"

    def fake_run(*args, **kwargs):
        # Simulate piper writing the output file
        out = kwargs["input"]
        path = Path(args[0][args[0].index("--output_file") + 1])
        path.write_bytes(b"RIFF")
        return MagicMock(returncode=0, stderr=b"")

    with patch("subprocess.run", side_effect=fake_run):
        result = tts.synthesize("Hello world")

    assert result.suffix == ".wav"
    assert result.exists()


def test_synthesize_missing_voice(tmp_path):
    tts = TTSEngine(piper_binary="piper", voice_path=tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError, match="voice model"):
        tts.synthesize("Hello")


def test_synthesize_piper_not_found(engine):
    tts, _ = engine
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError, match="Piper binary"):
            tts.synthesize("Hello")


def test_synthesize_piper_nonzero_exit(engine):
    tts, _ = engine
    with patch("subprocess.run", return_value=MagicMock(returncode=1, stderr=b"error")):
        with pytest.raises(RuntimeError, match="Piper failed"):
            tts.synthesize("Hello")


def test_synthesize_timeout(engine):
    tts, _ = engine
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("piper", 15)):
        with pytest.raises(RuntimeError, match="timed out"):
            tts.synthesize("Hello")


def test_delete_removes_file(tmp_path):
    f = tmp_path / "audio.wav"
    f.write_bytes(b"data")
    TTSEngine.delete(f)
    assert not f.exists()


def test_delete_missing_file_no_error(tmp_path):
    TTSEngine.delete(tmp_path / "nonexistent.wav")  # must not raise
