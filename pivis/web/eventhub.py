"""Fan out audio + detection events from the shared queues to per-SSE-client queues.

Persistent reader tasks drain the source queues once and copy each event to every
subscriber, so no consumer cancels a half-completed get() (which previously dropped
the rare audio greeting amid the detection flood), and multiple browser tabs each
get their own full copy of the stream.
"""
import asyncio

from pivis.state import AudioEvent


class EventHub:
    def __init__(self, events: asyncio.Queue, detections: asyncio.Queue, maxsize: int = 256) -> None:
        self._sources = (events, detections)
        self._maxsize = maxsize
        self._subs: set[asyncio.Queue] = set()
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._pump(src)) for src in self._sources]

    async def _pump(self, src: asyncio.Queue) -> None:
        while True:
            item = await src.get()
            for q in list(self._subs):
                if q.full():
                    _drop_oldest_non_audio(q)
                q.put_nowait(item)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._tasks = []


def _drop_oldest_non_audio(q: asyncio.Queue) -> None:
    """Make room in a full subscriber queue, preferring to drop a detection over a
    greeting so audio is never lost to backpressure."""
    buf = []
    while not q.empty():
        buf.append(q.get_nowait())
    for i, item in enumerate(buf):
        if not isinstance(item, AudioEvent):
            del buf[i]
            break
    else:
        if buf:
            buf.pop(0)  # all audio (unlikely) — drop the oldest
    for item in buf:
        q.put_nowait(item)
