"""Audio processor for real-time framing and silence detection."""

import logging
import struct
from typing import Optional, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process audio streams: framing, silence detection, phrase segmentation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        silence_threshold: float = 0.02,
        min_silence_duration_ms: int = 600,
    ):
        """Initialize audio processor.

        Args:
            sample_rate: Sample rate in Hz (default: 16kHz)
            frame_duration_ms: Frame size in milliseconds
            silence_threshold: RMS amplitude threshold for silence detection
            min_silence_duration_ms: Minimum silence duration to consider phrase
                boundary. Was 300ms; raised after user reports of long
                sentences getting split into several segments -- normal
                mid-sentence pauses (breaths, commas, brief hesitation)
                routinely exceed 300ms, so most of those were being treated
                as full sentence boundaries. Now that faster-whisper cut
                inference to ~0.6-0.9s/phrase (down from ~2s), the
                wait-for-silence portion is the bigger share of total
                latency, so there's more room to raise this without hurting
                perceived responsiveness.
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.silence_threshold = silence_threshold
        self.min_silence_duration_ms = min_silence_duration_ms

        # Calculate frame size in samples
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.bytes_per_sample = 2  # 16-bit audio

        # State tracking. silence_duration_ms starts "pre-satisfied" at the
        # threshold: a fresh stream has no prior audio, so it's reasonable to
        # treat it as already-silent. Starting at 0 meant the very first
        # utterance after connecting could never set in_phrase=True (that
        # only happens when silence_duration_ms >= min_silence_duration_ms
        # *before* speech starts), so it silently never completed as a
        # phrase — confirmed live: tone-then-silence right after connect
        # produced zero server-side phrase_complete events.
        self.buffer = bytearray()
        self.silence_duration_ms = min_silence_duration_ms
        self.in_phrase = False
        self.current_phrase = bytearray()
        self.frame_count = 0

    def process_frame(self, audio_data: bytes, frame_id: str) -> Optional[dict]:
        """Process a single audio frame.

        Args:
            audio_data: Raw PCM audio bytes (16-bit)
            frame_id: Frame identifier

        Returns:
            Processed frame dict with segments, or None if incomplete
        """
        if not audio_data:
            return None

        try:
            # Add to buffer
            self.buffer.extend(audio_data)
            self.frame_count += 1

            # Extract complete frames from buffer
            frames = []
            while len(self.buffer) >= self.frame_size * self.bytes_per_sample:
                frame_bytes = bytes(
                    self.buffer[: self.frame_size * self.bytes_per_sample]
                )
                del self.buffer[: self.frame_size * self.bytes_per_sample]

                rms = self._calculate_rms(frame_bytes)
                is_silence = rms < self.silence_threshold

                result = self._process_audio_frame(
                    frame_bytes, rms, is_silence, frame_id
                )
                if result:
                    frames.append(result)

            return {
                "frame_id": frame_id,
                "frame_count": self.frame_count,
                "frames": frames,
                "buffer_size": len(self.buffer),
            }

        except Exception as e:
            logger.error(f"Frame {frame_id}: processing error - {e}")
            return None

    def _process_audio_frame(
        self,
        frame_bytes: bytes,
        rms: float,
        is_silence: bool,
        frame_id: str,
    ) -> Optional[dict]:
        """Process a single 20ms frame.

        Args:
            frame_bytes: PCM audio bytes
            rms: RMS amplitude
            is_silence: Whether frame is silence
            frame_id: Original frame ID

        Returns:
            Processed frame dict or None
        """
        if is_silence:
            self.silence_duration_ms += self.frame_duration_ms

            # Check if silence duration exceeds threshold (phrase boundary)
            if (
                self.silence_duration_ms >= self.min_silence_duration_ms
                and self.in_phrase
            ):
                # End current phrase
                self.in_phrase = False
                if self.current_phrase:
                    result = {
                        "type": "phrase_complete",
                        "audio_data": bytes(self.current_phrase),
                        "duration_ms": len(self.current_phrase)
                        // (self.bytes_per_sample * self.sample_rate // 1000),
                        "frame_id": frame_id,
                    }
                    self.current_phrase = bytearray()
                    return result
        else:
            # Non-silence frame
            if self.silence_duration_ms >= self.min_silence_duration_ms:
                # Transitioning from silence to speech
                self.in_phrase = True

            self.silence_duration_ms = 0
            self.current_phrase.extend(frame_bytes)

        return {
            "type": "audio_frame",
            "audio_data": frame_bytes,
            "rms": rms,
            "is_silence": is_silence,
            "in_phrase": self.in_phrase,
            "frame_id": frame_id,
        }

    def _calculate_rms(self, audio_bytes: bytes) -> float:
        """Calculate RMS (Root Mean Square) amplitude.

        Args:
            audio_bytes: 16-bit PCM audio bytes

        Returns:
            RMS amplitude (0.0 to 1.0 normalized)
        """
        if len(audio_bytes) == 0:
            return 0.0

        try:
            # Unpack 16-bit signed samples
            samples = struct.unpack(
                f"<{len(audio_bytes) // 2}h",  # Little-endian signed shorts
                audio_bytes,
            )

            # Calculate RMS
            sum_squares = sum(s ** 2 for s in samples)
            rms = (sum_squares / len(samples)) ** 0.5

            # Normalize to 0-1 range (16-bit max is 32768)
            return rms / 32768.0

        except Exception as e:
            logger.error(f"RMS calculation error: {e}")
            return 0.0

    def flush_phrase(self) -> Optional[dict]:
        """Flush any remaining phrase buffer (e.g., at stream end).

        Returns:
            Final phrase dict or None
        """
        if self.current_phrase:
            result = {
                "type": "phrase_complete",
                "audio_data": bytes(self.current_phrase),
                "duration_ms": len(self.current_phrase)
                // (self.bytes_per_sample * self.sample_rate // 1000),
                "final": True,
            }
            self.current_phrase = bytearray()
            return result
        return None

    def reset(self) -> None:
        """Reset processor state."""
        self.buffer = bytearray()
        self.silence_duration_ms = self.min_silence_duration_ms
        self.in_phrase = False
        self.current_phrase = bytearray()
        self.frame_count = 0
