"""Audio processing module for PiVis.

Real-time audio stream ingestion, processing, and LLM integration.
"""

from pivis.audio.stream_handler import AudioStreamHandler
from pivis.audio.processor import AudioProcessor
from pivis.audio.stt import STTService, WhisperSTT
from pivis.audio.llm_handler import LLMService, LocalLLM, OpenAILLM
from pivis.audio.resilience import CircuitBreaker, RetryPolicy, ErrorAccumulator

__all__ = [
    "AudioStreamHandler",
    "AudioProcessor",
    "STTService",
    "WhisperSTT",
    "LLMService",
    "LocalLLM",
    "OpenAILLM",
    "CircuitBreaker",
    "RetryPolicy",
    "ErrorAccumulator",
]
