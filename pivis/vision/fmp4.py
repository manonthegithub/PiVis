import asyncio
import logging

logger = logging.getLogger(__name__)

def _build_ffmpeg_cmd(fps: int = 20) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-fflags", "+nobuffer", "-flags", "low_delay",
        # use_wallclock_as_timestamps: stamp each packet with real wall clock so
        # the MP4 muxer always receives valid, monotonically-increasing PTS from
        # raw H264 pipe input (which carries no timing in the bitstream).
        "-use_wallclock_as_timestamps", "1",
        "-r", str(fps), "-f", "h264", "-i", "pipe:0",
        "-c:v", "copy",
        "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
        # Emit a fragment every 100ms (not only at keyframes) and flush each
        # packet immediately so the browser gets low-latency chunks.
        "-frag_duration", "100000", "-flush_packets", "1",
        "pipe:1",
    ]
_READ_SIZE = 65_536


class FMP4Streamer:
    """Wraps an ffmpeg subprocess that remuxes raw H.264 → fragmented MP4."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self, fps: int = 20) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *_build_ffmpeg_cmd(fps),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        asyncio.create_task(self._log_stderr())
        logger.info("FMP4Streamer: ffmpeg started (pid %d)", self._proc.pid)

    async def _log_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        async for line in self._proc.stderr:
            logger.warning("ffmpeg: %s", line.decode().rstrip())

    async def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
            await asyncio.wait_for(self._proc.wait(), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            self._proc.kill()
        self._proc = None
        logger.info("FMP4Streamer: ffmpeg stopped")

    async def push(self, nal_bytes: bytes) -> None:
        if self._proc and self._proc.stdin and not self._proc.stdin.is_closing():
            self._proc.stdin.write(nal_bytes)
            await self._proc.stdin.drain()

    async def read_chunk(self) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            return b""
        try:
            return await self._proc.stdout.read(_READ_SIZE)
        except Exception:
            return b""


async def run_fmp4_loop(
    nal_queue: asyncio.Queue,
    fmp4_queue: asyncio.Queue,
    fps: int = 20,
    queues=None,  # Queues instance for caching init segment
) -> None:
    """Read NAL units from nal_queue, remux via ffmpeg, push fMP4 chunks to fmp4_queue."""
    streamer = FMP4Streamer()
    await streamer.start(fps)
    chunks_produced = 0

    async def _reader() -> None:
        nonlocal chunks_produced
        while True:
            chunk = await streamer.read_chunk()
            if not chunk:
                break
            chunks_produced += 1
            if chunks_produced == 1:
                logger.info("fMP4: first chunk produced (%d bytes)", len(chunk))
                # Cache init segment so new WebSocket clients get it on reconnect
                if queues is not None:
                    queues.fmp4_init = chunk
            await fmp4_queue.put(chunk)

    reader_task = asyncio.create_task(_reader())
    nals_received = 0
    try:
        while True:
            nal_bytes, _keyframe, _pts = await nal_queue.get()
            nals_received += 1
            if nals_received == 1:
                logger.info("fMP4: first NAL unit received (%d bytes)", len(nal_bytes))
            await streamer.push(nal_bytes)
    except asyncio.CancelledError:
        pass
    finally:
        reader_task.cancel()
        await streamer.stop()
