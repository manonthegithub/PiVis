import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pivis.state import AudioEvent, Queues
from pivis.vision.audio import BothAudioOutput, BrowserAudioOutput, LocalAudioOutput, make_audio_output


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "test.wav"
    p.write_bytes(b"RIFF")
    return p


@pytest.mark.asyncio
async def test_browser_pushes_event(wav):
    q = Queues()
    out = BrowserAudioOutput()
    await out.play(wav, "Hello!", q.events)
    event = q.events.get_nowait()
    assert event.wav_url == f"/audio/{wav.name}"
    assert event.text == "Hello!"


@pytest.mark.asyncio
async def test_local_calls_aplay(wav):
    proc_mock = AsyncMock()
    proc_mock.returncode = 0
    proc_mock.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", return_value=proc_mock):
        out = LocalAudioOutput(device="default")
        q = Queues()
        await out.play(wav, "Hello!", q.events)

    assert not wav.exists()  # deleted after play


@pytest.mark.asyncio
async def test_local_aplay_not_found_no_crash(wav):
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        out = LocalAudioOutput()
        q = Queues()
        await out.play(wav, "Hello!", q.events)  # must not raise


@pytest.mark.asyncio
async def test_both_runs_browser_and_local(wav):
    q = Queues()
    proc_mock = AsyncMock()
    proc_mock.returncode = 0
    proc_mock.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", return_value=proc_mock):
        out = BothAudioOutput(device="default")
        await out.play(wav, "Hi!", q.events)

    event = q.events.get_nowait()
    assert event.text == "Hi!"


def test_make_audio_output_modes():
    assert isinstance(make_audio_output("browser"), BrowserAudioOutput)
    assert isinstance(make_audio_output("local"), LocalAudioOutput)
    assert isinstance(make_audio_output("both"), BothAudioOutput)


def test_make_audio_output_invalid():
    with pytest.raises(ValueError, match="Unknown audio_output"):
        make_audio_output("hdmi")
