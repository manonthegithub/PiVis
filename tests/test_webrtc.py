import asyncio

import av
import pytest

from pivis.vision.webrtc import NalHub, PreEncodedH264Track

_IDR = b"\x00\x00\x00\x01\x67\x42\x00\x1e" + b"\x00\x00\x00\x01\x65" + b"\x00" * 32
_P = b"\x00\x00\x00\x01\x41" + b"\x00" * 16


@pytest.mark.asyncio
async def test_hub_fans_out_to_subscribers():
    nal_queue: asyncio.Queue = asyncio.Queue()
    hub = NalHub(nal_queue)
    hub.start()
    a, b = hub.subscribe(), hub.subscribe()
    await nal_queue.put((_IDR, True, 0))
    await nal_queue.put((_P, False, 3000))
    assert (await asyncio.wait_for(a.get(), 1))[2] == 0
    assert (await asyncio.wait_for(b.get(), 1))[2] == 0
    assert (await asyncio.wait_for(a.get(), 1))[2] == 3000
    await hub.stop()


@pytest.mark.asyncio
async def test_new_subscriber_misses_earlier_units():
    nal_queue: asyncio.Queue = asyncio.Queue()
    hub = NalHub(nal_queue)
    hub.start()
    early = hub.subscribe()
    await nal_queue.put((_IDR, True, 0))
    await asyncio.wait_for(early.get(), 1)  # drain so the hub has processed it
    late = hub.subscribe()
    await nal_queue.put((_P, False, 3000))
    assert (await asyncio.wait_for(late.get(), 1))[2] == 3000  # only the new unit
    await hub.stop()


@pytest.mark.asyncio
async def test_track_waits_for_keyframe_then_yields_packet():
    nal_queue: asyncio.Queue = asyncio.Queue()
    hub = NalHub(nal_queue)
    hub.start()
    track = PreEncodedH264Track(hub)
    await nal_queue.put((_P, False, 1000))   # non-keyframe first — must be skipped
    await nal_queue.put((_IDR, True, 2000))  # keyframe — playback starts here
    packet = await asyncio.wait_for(track.recv(), 1)
    assert isinstance(packet, av.Packet)
    assert packet.pts == 2000
    track.stop()
    await hub.stop()
