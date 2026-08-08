import asyncio
from dataclasses import dataclass, field

EventQueue = asyncio.Queue  # type alias for annotation use


@dataclass
class AudioEvent:
    wav_url: str
    text: str


@dataclass
class AppState:
    has_person: bool = False
    last_greeting_at: float = 0.0
    sse_client_count: int = 0
    latest_jpeg: bytes | None = None
    latest_side_jpeg: bytes | None = None


@dataclass
class Queues:
    frames: asyncio.Queue[bytes] = field(default_factory=lambda: asyncio.Queue(maxsize=2))
    events: asyncio.Queue[AudioEvent] = field(default_factory=asyncio.Queue)
    controls: asyncio.Queue[dict] = field(default_factory=asyncio.Queue)


LIGHTING_PRESETS = {
    "daylight": {"AnalogueGain": 1.0},
    "indoor":   {"AnalogueGain": 4.0},
    "dim":      {"AnalogueGain": 8.0},
}
