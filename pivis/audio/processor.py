"""Audio processor for real-time framing and speech/silence segmentation."""

import logging
from typing import Optional, List, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD's native chunk size at 16kHz -- the model's internal windowing
# is calibrated for exactly this many samples per call; using a different
# size isn't guaranteed to produce a correctly-calibrated probability.
VAD_CHUNK_SAMPLES = 512


def _default_vad_model():
    """Lazily imports and returns the shared Silero VAD instance.

    faster_whisper.vad.get_vad_model() is itself @functools.lru_cache'd, so
    this is the same singleton stt.py's WhisperSTT would otherwise load a
    second time via transcribe(vad_filter=True) -- using it here for the
    actual segmentation decision costs nothing extra to load, and means
    vad_filter is turned back off in stt.py to avoid running VAD twice on
    the same audio.
    """
    from faster_whisper.vad import get_vad_model

    return get_vad_model()


class AudioProcessor:
    """Process audio streams: framing, VAD-based phrase segmentation.

    Segmentation used to be a crude RMS-amplitude threshold, which
    couldn't distinguish real speech from any other loud sound -- confirmed
    live: a pure sine-wave test tone reliably fooled it into starting a
    "phrase", which whisper then hallucinated real-sounding text for
    ("Thanks for watching!", "You", etc, each with deceptively high
    confidence). Now uses the same Silero VAD model faster-whisper's own
    vad_filter uses internally, for the segmentation decision itself
    rather than as a late double-check after a phrase is already cut.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 32,
        silence_threshold: float = 0.5,
        min_silence_duration_ms: int = 450,
        vad_model=None,
    ):
        """Initialize audio processor.

        Args:
            sample_rate: Sample rate in Hz. Must be 16000 -- Silero VAD's
                512-sample native chunk size only lines up with 32ms of
                audio at 16kHz, and faster-whisper/Whisper itself hardcode
                16kHz everywhere else in this pipeline too.
            frame_duration_ms: Frame size in ms. Must be 32 (512 samples @
                16kHz = VAD_CHUNK_SAMPLES) to match Silero VAD's native
                chunk size -- don't change without also revisiting
                VAD_CHUNK_SAMPLES.
            silence_threshold: VAD speech-probability threshold (0-1, not
                an RMS amplitude like before). 0.5 matches faster-whisper's
                own VadOptions default.
            min_silence_duration_ms: Minimum silence duration to consider
                phrase boundary. This default is only used if
                AudioProcessor is constructed without an explicit value --
                audio_app.py always passes MIN_SILENCE_DURATION_MS from
                the environment, which is the actual live value and the
                one worth tuning, not this default. History: 300ms split
                long sentences into too many segments (normal mid-sentence
                pauses routinely exceed 300ms); 600ms fixed that but was
                reported to hurt accuracy; 450ms is a middle ground.
            vad_model: Injectable for testing -- a callable matching
                Silero's `model(float32_samples, num_samples=N) ->
                np.ndarray` of per-chunk speech probabilities. Defaults to
                the real shared model; tests substitute a fake so behavior
                doesn't depend on a real neural net's judgment call on
                synthetic (non-speech-shaped) test audio.
        """
        if sample_rate != 16000:
            raise ValueError("AudioProcessor requires sample_rate=16000 (Silero VAD constraint)")
        if frame_duration_ms * sample_rate // 1000 != VAD_CHUNK_SAMPLES:
            raise ValueError(
                f"frame_duration_ms={frame_duration_ms} at sample_rate={sample_rate} must "
                f"work out to VAD_CHUNK_SAMPLES={VAD_CHUNK_SAMPLES} samples"
            )
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.silence_threshold = silence_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self._vad_model = vad_model or _default_vad_model()

        # Calculate frame size in samples
        self.frame_size = VAD_CHUNK_SAMPLES
        self.bytes_per_sample = 2  # 16-bit audio

        # State tracking. silence_duration_ms starts "pre-satisfied" at the
        # threshold: a fresh stream has no prior audio, so it's reasonable
        # to treat it as already-silent. Starting at 0 meant the very
        # first utterance after connecting could never set in_phrase=True
        # (that only happens when silence_duration_ms >=
        # min_silence_duration_ms *before* speech starts), so it silently
        # never completed as a phrase — confirmed live: tone-then-silence
        # right after connect produced zero server-side phrase_complete
        # events.
        self.buffer = bytearray()
        self.silence_duration_ms = min_silence_duration_ms
        self.in_phrase = False
        self.current_phrase = bytearray()
        self.frame_count = 0

    def process_frame(self, audio_data: bytes, frame_id: str) -> Optional[dict]:
        """Process an incoming audio chunk.

        Args:
            audio_data: Raw PCM audio bytes (16-bit)
            frame_id: Frame identifier

        Returns:
            Processed frame dict with any phrase_complete events, or None
            on error.
        """
        if not audio_data:
            return None

        try:
            self.buffer.extend(audio_data)
            self.frame_count += 1

            chunk_bytes = self.frame_size * self.bytes_per_sample
            n_complete = len(self.buffer) // chunk_bytes
            frames: List[dict] = []

            if n_complete > 0:
                take = n_complete * chunk_bytes
                batch_bytes = bytes(self.buffer[:take])
                del self.buffer[:take]

                # One batched VAD call per incoming chunk (not one call per
                # 512-sample sub-chunk) so Silero's internal cross-chunk
                # context rolling (see faster_whisper.vad.SileroVADModel)
                # actually has neighboring audio to roll from, rather than
                # each call starting from a cold h/c state.
                samples = np.frombuffer(batch_bytes, dtype="<i2").astype(np.float32) / 32768.0
                probs = self._vad_model(samples, num_samples=self.frame_size)

                for i in range(n_complete):
                    sub_bytes = batch_bytes[i * chunk_bytes : (i + 1) * chunk_bytes]
                    prob = float(probs[i])
                    is_silence = prob < self.silence_threshold
                    result = self._process_audio_frame(sub_bytes, prob, is_silence, frame_id)
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
        speech_prob: float,
        is_silence: bool,
        frame_id: str,
    ) -> Optional[dict]:
        """Process a single VAD chunk (32ms / 512 samples @ 16kHz).

        Args:
            frame_bytes: PCM audio bytes
            speech_prob: VAD speech probability for this chunk (0-1)
            is_silence: Whether this chunk is below silence_threshold
            frame_id: Original frame ID

        Returns:
            A "phrase_complete" dict when this frame closes out a phrase,
            else None. process_frame's only caller (_on_frame in
            audio_app.py) discards anything that isn't phrase_complete, so
            building a dict for every chunk regardless would be wasted
            allocation on the main event loop -- see git history for the
            prior "audio_frame" version of this method.
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

        return None

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
