"""Tests for audio module."""

import pytest
import asyncio
import struct
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch

import numpy as np

from pivis.audio import AudioStreamHandler, AudioProcessor
from pivis.audio.processor import VAD_CHUNK_SAMPLES
from pivis.audio.stt import WhisperSTT, STTService
from pivis.audio.llm_handler import LocalLLM, LLMService
from pivis.audio.resilience import CircuitBreaker, CircuitBreakerConfig, RetryPolicy


class FakeVAD:
    """Stand-in for the real Silero VAD model in tests: classifies any
    non-zero chunk as speech (high probability), any all-zero chunk as
    silence (low probability). Neither faster-whisper nor onnxruntime are
    installed in this test environment, and even where they are, tests
    should control classification deterministically rather than depend on
    a real neural net's judgment call on synthetic (non-speech-shaped)
    test audio -- AudioProcessor's vad_model= constructor param exists
    for exactly this."""

    def __call__(self, samples: np.ndarray, num_samples: int = VAD_CHUNK_SAMPLES) -> np.ndarray:
        n_chunks = len(samples) // num_samples
        probs = []
        for i in range(n_chunks):
            chunk = samples[i * num_samples : (i + 1) * num_samples]
            probs.append(0.9 if np.abs(chunk).max() > 1e-6 else 0.05)
        return np.array(probs)


def speech_chunk_bytes(amplitude: int = 10000) -> bytes:
    """One VAD-chunk-sized (512 samples @ 16kHz = 32ms) constant-amplitude
    "speech" PCM chunk -- FakeVAD classifies any non-zero chunk as speech,
    so the actual waveform shape doesn't matter here, just non-zero."""
    return struct.pack(f"<{VAD_CHUNK_SAMPLES}h", *([amplitude] * VAD_CHUNK_SAMPLES))


def silence_chunk_bytes() -> bytes:
    """One VAD-chunk-sized all-zero PCM chunk -- FakeVAD classifies this
    as silence."""
    return struct.pack(f"<{VAD_CHUNK_SAMPLES}h", *([0] * VAD_CHUNK_SAMPLES))


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = AudioProcessor(
            sample_rate=16000,
            frame_duration_ms=32,
            silence_threshold=0.5,
            vad_model=FakeVAD(),
        )

        assert processor.sample_rate == 16000
        assert processor.frame_duration_ms == 32
        assert processor.silence_threshold == 0.5
        assert processor.frame_size == VAD_CHUNK_SAMPLES

    def test_rejects_non_16khz_or_mismatched_frame_duration(self):
        """Regression guard: Silero VAD's 512-sample native chunk size
        only lines up with 32ms at 16kHz. A silent mismatch here would
        make process_frame feed VAD incorrectly-sized/calibrated input."""
        with pytest.raises(ValueError):
            AudioProcessor(sample_rate=8000, vad_model=FakeVAD())
        with pytest.raises(ValueError):
            AudioProcessor(frame_duration_ms=20, vad_model=FakeVAD())

    def test_vad_classifies_speech_vs_silence(self):
        """Test that VAD probability (not RMS amplitude) drives the
        speech/silence classification that in_phrase transitions on."""
        processor = AudioProcessor(vad_model=FakeVAD())

        result = processor.process_frame(speech_chunk_bytes(), "frame_1")
        assert result is not None
        # A single speech chunk isn't 450ms of trailing silence yet, so no
        # phrase_complete, but in_phrase should now be True.
        assert processor.in_phrase

        processor2 = AudioProcessor(vad_model=FakeVAD())
        processor2.process_frame(silence_chunk_bytes(), "frame_1")
        assert not processor2.in_phrase

    def test_frame_processing(self):
        """Test frame processing and buffering."""
        processor = AudioProcessor(vad_model=FakeVAD())

        result = processor.process_frame(speech_chunk_bytes(), "frame_1")

        assert result is not None
        assert result["frame_id"] == "frame_1"
        assert result["frame_count"] == 1

    def test_processor_reset(self):
        """Test processor state reset."""
        processor = AudioProcessor(vad_model=FakeVAD())

        processor.process_frame(speech_chunk_bytes(), "frame_1")
        processor.reset()

        assert processor.frame_count == 0
        assert len(processor.buffer) == 0
        assert not processor.in_phrase

    def test_first_utterance_completes_without_leading_silence(self):
        """Regression test: a fresh processor used to require >=300ms of
        silence *before* the first utterance to ever set in_phrase=True
        (silence_duration_ms started at 0, not the threshold), so speaking
        immediately after connecting never produced a phrase_complete event
        at all. Confirmed live via a real WebSocket test against the
        deployed audio module before this fix."""
        processor = AudioProcessor(min_silence_duration_ms=300, vad_model=FakeVAD())
        speech_frame = speech_chunk_bytes()
        silence_frame = silence_chunk_bytes()

        got_phrase = False
        for i in range(7):  # ~224ms of speech, no silence beforehand
            result = processor.process_frame(speech_frame, f"speech-{i}")
            got_phrase |= any(f["type"] == "phrase_complete" for f in result["frames"])
        for i in range(12):  # >300ms of trailing silence closes the phrase
            result = processor.process_frame(silence_frame, f"silence-{i}")
            got_phrase |= any(f["type"] == "phrase_complete" for f in result["frames"])

        assert got_phrase, "first utterance after connecting never completed as a phrase"


class TestSTTService:
    """Tests for STT service."""

    @pytest.mark.asyncio
    async def test_stt_service_initialization(self):
        """Test STT service initialization."""
        service = STTService()
        assert service.backend is not None
        assert service.retry_attempts == 3

    @pytest.mark.asyncio
    async def test_transcribe_phrase_success(self):
        """Test successful transcription."""
        # Mock backend
        mock_backend = AsyncMock()
        mock_backend.transcribe.return_value = {
            "text": "hello world",
            "confidence": 0.95,
            "language": "en",
        }

        service = STTService(backend=mock_backend)
        audio_bytes = b"\x00" * 16000  # 1 second of silence

        result = await service.transcribe_phrase(audio_bytes)

        assert result["text"] == "hello world"
        assert result["confidence"] == 0.95
        mock_backend.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_phrase_with_retry(self):
        """Test transcription with retry logic."""
        mock_backend = AsyncMock()
        # First call fails, second succeeds
        mock_backend.transcribe.side_effect = [
            {"text": "", "confidence": 0.0, "language": "en", "error": "Timeout"},
            {
                "text": "hello",
                "confidence": 0.9,
                "language": "en",
            },
        ]

        service = STTService(backend=mock_backend)
        result = await service.transcribe_phrase(b"\x00" * 16000)

        assert result["text"] == "hello"
        assert mock_backend.transcribe.call_count == 2

    @pytest.mark.asyncio
    async def test_whisper_local_handles_no_speech_segments(self):
        """Regression test: faster-whisper (like openai-whisper before it)
        returns an empty segments sequence -- not missing, just empty --
        whenever a phrase has no detected speech. `segments[0]` without a
        guard raised IndexError on every such phrase before this fix
        (confirmed live against the openai-whisper backend originally;
        preserved here for the current faster-whisper backend)."""
        stt = WhisperSTT.__new__(WhisperSTT)  # skip __init__'s real model load
        stt.model_name = "tiny"
        stt.api_key = None
        stt._infer_lock = asyncio.Lock()
        stt.beam_size = 5
        stt.local_model = Mock()
        # faster-whisper returns (segments_generator, info), not a dict.
        stt.local_model.transcribe = Mock(return_value=(iter([]), SimpleNamespace(language="en")))

        audio_bytes = struct.pack("<160h", *[0] * 160)
        result = await stt.transcribe(audio_bytes)

        assert result.get("error") is None
        assert result["text"] == ""
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_transcribe_calls_are_serialized(self):
        """Regression test: whisper's model isn't safe for concurrent
        .transcribe() calls from multiple threads. Once _on_frame started
        running each phrase's transcription as its own background task,
        overlapping phrases could call the shared model concurrently --
        confirmed live: two calls landing ~16ms apart produced corrupted
        internal state (a zero-element tensor reshape error, then a raw
        nn.Linear object surfacing as an error message) and the pod was
        OOMKilled shortly after from concurrent inferences stacking up.
        _infer_lock must keep calls to the shared model serialized even
        when multiple transcribe() coroutines are in flight at once."""
        stt = WhisperSTT.__new__(WhisperSTT)
        stt.model_name = "tiny"
        stt.api_key = None
        stt._infer_lock = asyncio.Lock()
        stt.beam_size = 5
        stt.local_model = Mock()

        concurrent_calls = 0
        max_concurrent = 0

        def fake_transcribe(audio_data, **kwargs):
            nonlocal concurrent_calls, max_concurrent
            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)
            import time

            time.sleep(0.05)  # simulate real inference taking a moment
            concurrent_calls -= 1
            segment = SimpleNamespace(text="hi", no_speech_prob=0.1)
            return iter([segment]), SimpleNamespace(language="en")

        stt.local_model.transcribe = fake_transcribe

        audio_bytes = struct.pack("<160h", *[100] * 160)
        await asyncio.gather(
            stt.transcribe(audio_bytes),
            stt.transcribe(audio_bytes),
            stt.transcribe(audio_bytes),
        )

        assert max_concurrent == 1, (
            f"expected transcribe() calls to be serialized, but "
            f"{max_concurrent} ran concurrently"
        )


class TestLLMService:
    """Tests for LLM service."""

    @pytest.mark.asyncio
    async def test_llm_service_initialization(self):
        """Test LLM service initialization."""
        service = LLMService()
        assert service.backend is not None

    @pytest.mark.asyncio
    async def test_process_transcription_success(self):
        """Test successful LLM processing."""
        mock_backend = AsyncMock()
        mock_backend.process_text.return_value = {
            "response": "That's interesting!",
            "tokens_used": 50,
            "model": "test-model",
        }

        service = LLMService(backend=mock_backend)
        transcription = {"text": "hello", "confidence": 0.95, "language": "en"}

        result = await service.process_transcription(transcription)

        assert result["response"] == "That's interesting!"
        assert result["input"] == transcription

    @pytest.mark.asyncio
    async def test_process_empty_transcription(self):
        """Test handling of empty transcription."""
        service = LLMService()
        transcription = {"text": "", "confidence": 0.0, "language": "en"}

        result = await service.process_transcription(transcription)

        assert result["error"] == "Empty input text"


class TestCircuitBreaker:
    """Tests for circuit breaker."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker("test_service")
        mock_func = AsyncMock(return_value="success")

        result = await breaker.call(mock_func)

        assert result == "success"
        assert breaker.state.value == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test_service", config)
        mock_func = AsyncMock(side_effect=Exception("Service down"))

        # Trigger failures
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(mock_func)

        assert breaker.state.value == "open"

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test_service", config)
        mock_func = AsyncMock(side_effect=Exception("Service down"))

        # Open the circuit
        with pytest.raises(Exception):
            await breaker.call(mock_func)

        # Try to call while open
        with pytest.raises(Exception, match="Circuit test_service is OPEN"):
            await breaker.call(mock_func)


class TestRetryPolicy:
    """Tests for retry policy."""

    def test_retry_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(initial_delay_ms=100, max_delay_ms=5000)

        delay0 = policy.calculate_delay(0)
        delay1 = policy.calculate_delay(1)
        delay2 = policy.calculate_delay(2)

        assert delay0 == 100
        assert delay1 == 200
        assert delay2 == 400

    @pytest.mark.asyncio
    async def test_retry_success_on_second_attempt(self):
        """Test successful retry on second attempt."""
        policy = RetryPolicy(max_retries=3, initial_delay_ms=10)
        mock_func = AsyncMock(side_effect=[Exception("Failed"), "success"])

        result = await policy.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry exhaustion."""
        policy = RetryPolicy(max_retries=2, initial_delay_ms=10)
        mock_func = AsyncMock(side_effect=Exception("Always fails"))

        with pytest.raises(Exception, match="Always fails"):
            await policy.execute(mock_func)

        assert mock_func.call_count == 3  # Initial + 2 retries


class TestAudioStreamHandler:
    """Tests for AudioStreamHandler."""

    @pytest.mark.asyncio
    async def test_stream_handler_initialization(self):
        """Test stream handler initialization."""
        handler = AudioStreamHandler(buffer_size=512, timeout_seconds=30)

        assert handler.buffer_size == 512
        assert handler.timeout_seconds == 30

    def test_frame_parsing(self):
        """Test audio frame parsing."""
        import json
        import base64

        handler = AudioStreamHandler()
        audio_data = b"\x00\x01\x02\x03"
        frame_json = json.dumps({
            "type": "audio_chunk",
            "timestamp": "2026-09-03T15:00:00Z",
            "audio_base64": base64.b64encode(audio_data).decode(),
            "sample_rate": 16000,
            "frame_id": "frame_1",
        })

        parsed = handler._parse_frame(frame_json, "stream_1")

        assert parsed is not None
        assert parsed["audio_data"] == audio_data
        assert parsed["sample_rate"] == 16000

    def test_get_active_streams(self):
        """Test getting active streams."""
        handler = AudioStreamHandler()
        handler.active_streams["stream_1"] = {"frame_count": 10}
        handler.active_streams["stream_2"] = {"frame_count": 20}

        streams = handler.get_active_streams()

        assert len(streams) == 2
        assert "stream_1" in streams
        assert "stream_2" in streams


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
