import asyncio
from dataclasses import dataclass, field

EventQueue = asyncio.Queue  # type alias for annotation use


@dataclass
class AudioEvent:
    wav_url: str
    text: str


@dataclass
class DetectionEvent:
    boxes: list[dict]       # [{x1,y1,x2,y2,confidence}, ...]
    has_person: bool
    sensor_timestamp_ns: int


@dataclass
class AppState:
    has_person: bool = False
    last_greeting_at: float = 0.0
    sse_client_count: int = 0


@dataclass
class Queues:
    events: asyncio.Queue[AudioEvent] = field(default_factory=asyncio.Queue)
    detections: asyncio.Queue[DetectionEvent] = field(default_factory=asyncio.Queue)
    controls: asyncio.Queue[dict] = field(default_factory=asyncio.Queue)
    nal_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    fmp4_queue: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)


LIGHTING_PRESETS = {
    "daylight": {"AnalogueGain": 1.0},
    "indoor":   {"AnalogueGain": 4.0},
    "dim":      {"AnalogueGain": 8.0},
}
