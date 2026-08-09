"""WebRTC backend: forwards the camera's pre-encoded H.264 to browsers via RTP.

No re-encoding — the NAL units picamera2 already produced are wrapped in av.Packet
and handed to aiortc's H.264 packetizer (RTCRtpSender.pack path).
"""
import asyncio
import logging
from fractions import Fraction

import av
from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

logger = logging.getLogger(__name__)

_VIDEO_TIME_BASE = Fraction(1, 90_000)  # RTP video clock; NAL pts already in 90kHz
_SUB_QUEUE_MAX = 120  # ~6s at 20fps; drop oldest if a peer stalls


class NalHub:
    """Drains the shared H.264 nal_queue and fans units out to per-peer queues."""

    def __init__(self, nal_queue: asyncio.Queue) -> None:
        self._nal_queue = nal_queue
        self._subs: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            unit = await self._nal_queue.get()  # (data, keyframe, pts_90k)
            for q in list(self._subs):
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(unit)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_SUB_QUEUE_MAX)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


class PreEncodedH264Track(MediaStreamTrack):
    """Yields the camera's H.264 access units as av.Packet for aiortc to packetize."""

    kind = "video"

    def __init__(self, hub: NalHub) -> None:
        super().__init__()
        self._hub = hub
        self._queue = hub.subscribe()
        self._started = False

    async def recv(self) -> av.Packet:
        while True:
            data, keyframe, pts = await self._queue.get()
            # Wait for a keyframe before the first packet so the decoder can start.
            if not self._started:
                if not keyframe:
                    continue
                self._started = True
            packet = av.Packet(data)
            packet.pts = pts
            packet.time_base = _VIDEO_TIME_BASE
            return packet

    def stop(self) -> None:
        super().stop()
        self._hub.unsubscribe(self._queue)


def _prefer_h264(pc: RTCPeerConnection) -> None:
    caps = RTCRtpSender.getCapabilities("video")
    h264 = [c for c in caps.codecs if c.mimeType == "video/H264"]
    for t in pc.getTransceivers():
        if t.kind == "video" and h264:
            t.setCodecPreferences(h264)


async def handle_offer(hub: NalHub, pcs: set, offer: dict) -> dict:
    """Create a peer connection for an SDP offer and return the SDP answer."""
    pc = RTCPeerConnection()
    pcs.add(pc)
    track = PreEncodedH264Track(hub)
    pc.addTrack(track)
    _prefer_h264(pc)

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        if pc.connectionState in ("failed", "closed", "disconnected"):
            track.stop()
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def close_all(pcs: set) -> None:
    for pc in list(pcs):
        await pc.close()
    pcs.clear()
