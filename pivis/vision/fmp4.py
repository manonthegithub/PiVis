import asyncio
import logging

logger = logging.getLogger(__name__)

def _build_ffmpeg_cmd(fps: int = 20) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        # genpts: generate missing PTS so mp4 muxer gets proper timestamps
        # r: nominal fps for PTS spacing
        "-fflags", "+genpts", "-r", str(fps), "-f", "h264", "-i", "pipe:0",
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
