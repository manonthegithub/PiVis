"""Tests for audio module."""

import pytest
import asyncio
import struct
from unittest.mock import Mock, AsyncMock, patch

from pivis.audio import AudioStreamHandler, AudioProcessor
from pivis.audio.stt import WhisperSTT, STTService
from pivis.audio.llm_handler import LocalLLM, LLMService
from pivis.audio.resilience import CircuitBreaker, CircuitBreakerConfig, RetryPolicy


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = AudioProcessor(
            sample_rate=16000,
            frame_duration_ms=20,
            silence_threshold=0.02,
        )

        assert processor.sample_rate == 16000
        assert processor.frame_duration_ms == 20
        assert processor.silence_threshold == 0.02

    def test_rms_calculation(self):
        """Test RMS amplitude calculation."""
        processor = AudioProcessor()

        # Create test audio: 1000 Hz sine wave (non-silence)
        samples = struct.pack("<100h", *[int(32767 * 0.5) for _ in range(100)])
        rms = processor._calculate_rms(samples)

        assert rms > 0.4  # Should be significant amplitude
        assert rms < 0.6

    def test_silence_detection(self):
        """Test silence detection."""
        processor = AudioProcessor(silence_threshold=0.02)

        # Silent audio
        silent_audio = struct.pack("<160h", *[0] * 160)
        rms = processor._calculate_rms(silent_audio)

        assert rms < processor.silence_threshold

    def test_frame_processing(self):
        """Test frame processing and buffering."""
        processor = AudioProcessor()

        # Create test audio frame
        audio_data = struct.pack("<160h", *[100] * 160)
        result = processor.process_frame(audio_data, "frame_1")

        assert result is not None
        assert result["frame_id"] == "frame_1"
        assert result["frame_count"] == 1

    def test_processor_reset(self):
        """Test processor state reset."""
        processor = AudioProcessor()

        # Process some audio
        audio_data = struct.pack("<160h", *[100] * 160)
        processor.process_frame(audio_data, "frame_1")

        # Reset
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
        processor = AudioProcessor(frame_duration_ms=20, min_silence_duration_ms=300)
        speech_frame = struct.pack("<%dh" % processor.frame_size, *([10000] * processor.frame_size))
        silence_frame = struct.pack("<%dh" % processor.frame_size, *([0] * processor.frame_size))

        got_phrase = False
        for i in range(10):  # ~200ms of speech, no silence beforehand
            result = processor.process_frame(speech_frame, f"speech-{i}")
            got_phrase |= any(f["type"] == "phrase_complete" for f in result["frames"])
        for i in range(20):  # >300ms of trailing silence closes the phrase
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
        """Regression test: Whisper returns `segments: []` (present, just
        empty) rather than omitting the key whenever a phrase has no
        detected speech. `result.get("segments", [{}])[0]` only guards a
        *missing* key, so `[][0]` raised IndexError on every such phrase —
        confirmed live: a real phrase sent to the deployed audio module
        failed all 3 retries with "list index out of range" before this
        fix."""
        stt = WhisperSTT.__new__(WhisperSTT)  # skip __init__'s real model load
        stt.model_name = "tiny"
        stt.api_key = None
        stt.local_model = Mock()
        stt.local_model.transcribe = Mock(
            return_value={"text": "", "segments": [], "language": "en"}
        )

        audio_bytes = struct.pack("<160h", *[0] * 160)
        result = await stt.transcribe(audio_bytes)

        assert result.get("error") is None
        assert result["text"] == ""
        assert result["confidence"] == 0.0


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
