from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Camera
    stream_fps: int = 20
    camera_width: int = 800
    camera_height: int = 600
    camera_analogue_gain: float = 4.0  # 1.0=normal, higher=more sensitive (indoor: 4-8)

    # Detection
    detection_interval_ms: int = 200
    detection_confidence: float = 0.5
    detection_input_size: int = 320  # YOLO input resolution; 320=fast, 640=accurate
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
