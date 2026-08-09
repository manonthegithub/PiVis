import asyncio
import logging

logger = logging.getLogger(__name__)

_FFMPEG_CMD = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-f", "h264", "-i", "pipe:0",
    "-c:v", "copy",
    "-f", "mp4",
    "-movflags", "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
    "pipe:1",
]
_READ_SIZE = 65_536


class FMP4Streamer:
    """Wraps an ffmpeg subprocess that remuxes raw H.264 → fragmented MP4."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *_FFMPEG_CMD,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        logger.info("FMP4Streamer: ffmpeg started (pid %d)", self._proc.pid)

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

    def push(self, nal_bytes: bytes) -> None:
        if self._proc and self._proc.stdin and not self._proc.stdin.is_closing():
            self._proc.stdin.write(nal_bytes)

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
) -> None:
    """Read NAL units from nal_queue, remux via ffmpeg, push fMP4 chunks to fmp4_queue."""
    streamer = FMP4Streamer()
    await streamer.start()

    async def _reader() -> None:
        while True:
            chunk = await streamer.read_chunk()
            if not chunk:
                break
            await fmp4_queue.put(chunk)

    reader_task = asyncio.create_task(_reader())
    try:
        while True:
            nal_bytes, _keyframe, _pts = await nal_queue.get()
            streamer.push(nal_bytes)
    except asyncio.CancelledError:
        pass
    finally:
        reader_task.cancel()
        await streamer.stop()
