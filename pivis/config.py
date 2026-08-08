from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Camera
    stream_fps: int = 20
    camera_resolution: tuple[int, int] = (1280, 720)

    # Detection
    detection_interval_ms: int = 200
    detection_confidence: float = 0.5
    yolo_model_path: Path = Path("models/yolov8n.onnx")

    # Greeting
    greeting_cooldown_s: int = 30
    claude_model: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str

    # TTS
    tts_voice_path: Path = Path("models/en_US-lessac-medium.onnx")
    piper_binary: str = "piper"

    # Audio output: "browser" | "local" | "both"
    audio_output: str = "both"
    audio_device: str = "default"
