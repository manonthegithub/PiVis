import asyncio

import pytest

from pivis.state import AudioEvent, DetectionEvent
from pivis.web.eventhub import EventHub


def _det():
    return DetectionEvent(boxes=[], has_person=True, sensor_timestamp_ns=0)


async def _drain(q, n, timeout=1.0):
    out = []
    for _ in range(n):
        out.append(await asyncio.wait_for(q.get(), timeout))
    return out


@pytest.mark.asyncio
async def test_audio_survives_detection_flood():
    # Regression: the rare audio greeting must not be lost among many detections.
    events, dets = asyncio.Queue(), asyncio.Queue()
    hub = EventHub(events, dets)
    sub = hub.subscribe()
    hub.start()
    for _ in range(50):
        await dets.put(_det())
    await events.put(AudioEvent(wav_url="/audio/x.wav", text="Hi!"))
    for _ in range(50):
        await dets.put(_det())

    received = await _drain(sub, 101)
    audio = [e for e in received if isinstance(e, AudioEvent)]
    assert len(audio) == 1 and audio[0].text == "Hi!"
    await hub.stop()


@pytest.mark.asyncio
async def test_broadcasts_to_multiple_subscribers():
    events, dets = asyncio.Queue(), asyncio.Queue()
    hub = EventHub(events, dets)
    a, b = hub.subscribe(), hub.subscribe()
    hub.start()
    await events.put(AudioEvent(wav_url="/audio/x.wav", text="Hi!"))
    for sub in (a, b):
        item = await asyncio.wait_for(sub.get(), 1.0)
        assert isinstance(item, AudioEvent) and item.text == "Hi!"
    await hub.stop()


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    events, dets = asyncio.Queue(), asyncio.Queue()
    hub = EventHub(events, dets)
    sub = hub.subscribe()
    hub.start()
    hub.unsubscribe(sub)
    await events.put(AudioEvent(wav_url="/audio/x.wav", text="Hi!"))
    await asyncio.sleep(0.05)
    assert sub.empty()
    await hub.stop()


@pytest.mark.asyncio
async def test_backpressure_drops_detection_not_audio():
    events, dets = asyncio.Queue(), asyncio.Queue()
    hub = EventHub(events, dets, maxsize=4)
    sub = hub.subscribe()
    hub.start()
    await events.put(AudioEvent(wav_url="/audio/x.wav", text="keep me"))
    for _ in range(20):  # overflow the size-4 sub queue
        await dets.put(_det())
    await asyncio.sleep(0.1)
    items = []
    while not sub.empty():
        items.append(sub.get_nowait())
    assert any(isinstance(i, AudioEvent) for i in items)  # audio preserved
    await hub.stop()
