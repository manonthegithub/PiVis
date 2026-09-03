"""Audio stream handler for real-time audio ingestion via WebSocket."""

import asyncio
import logging
import json
from typing import Callable, Dict, Optional
from collections import deque
import base64

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class AudioStreamHandler:
    """Manages real-time audio stream ingestion from multiple sources.

    Handles:
    - WebSocket connections for audio streaming
    - Multiple concurrent streams (local, remote, browser)
    - Frame buffering and out-of-order handling
    - Timeout and error management
    """

    def __init__(
        self,
        frame_callback: Optional[Callable] = None,
        buffer_size: int = 1024,
        timeout_seconds: float = 30.0,
    ):
        """Initialize audio stream handler.

        Args:
            frame_callback: Async callable to process audio frames
            buffer_size: Maximum frames to buffer per stream
            timeout_seconds: WebSocket read timeout
        """
        self.frame_callback = frame_callback
        self.buffer_size = buffer_size
        self.timeout_seconds = timeout_seconds
        self.active_streams: Dict[str, dict] = {}

    async def handle_connection(
        self,
        websocket: WebSocket,
        stream_id: str,
    ) -> None:
        """Handle incoming WebSocket audio connection.

        Args:
            websocket: FastAPI WebSocket connection
            stream_id: Unique identifier for this stream

        Raises:
            WebSocketDisconnect: When connection closes
        """
        await websocket.accept()
        logger.info(f"Audio stream connected: {stream_id}")

        # Initialize stream state
        self.active_streams[stream_id] = {
            "websocket": websocket,
            "buffer": deque(maxlen=self.buffer_size),
            "frame_count": 0,
            "last_frame_time": None,
            "sample_rate": None,
        }

        try:
            while True:
                try:
                    # Receive audio frame
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self.timeout_seconds,
                    )

                    # Parse frame
                    frame = self._parse_frame(data, stream_id)
                    if not frame:
                        continue

                    # Update stream state
                    stream_state = self.active_streams[stream_id]
                    stream_state["buffer"].append(frame)
                    stream_state["frame_count"] += 1
                    stream_state["sample_rate"] = frame.get("sample_rate", 16000)

                    logger.debug(
                        f"Stream {stream_id}: received frame "
                        f"{frame.get('frame_id')} (buffer: {len(stream_state['buffer'])})"
                    )

                    # Process frame if callback provided
                    if self.frame_callback:
                        await self.frame_callback(stream_id, frame)

                except asyncio.TimeoutError:
                    logger.warning(f"Stream {stream_id}: read timeout")
                    await self._send_error(websocket, "Connection timeout")
                    break

                except json.JSONDecodeError as e:
                    logger.error(f"Stream {stream_id}: malformed frame - {e}")
                    await self._send_error(websocket, "Invalid frame format")
                    continue

        except WebSocketDisconnect:
            logger.info(f"Audio stream disconnected: {stream_id}")
        except Exception as e:
            logger.error(f"Stream {stream_id}: unexpected error - {e}")
        finally:
            # Cleanup
            if stream_id in self.active_streams:
                frame_count = self.active_streams[stream_id].get("frame_count", 0)
                del self.active_streams[stream_id]
                logger.info(f"Stream {stream_id} closed ({frame_count} frames)")

    def _parse_frame(self, data: str, stream_id: str) -> Optional[dict]:
        """Parse incoming audio frame.

        Expected JSON format:
        {
            "type": "audio_chunk",
            "timestamp": "2026-09-03T14:00:00Z",
            "audio_base64": "AQIDBA==",
            "sample_rate": 16000,
            "frame_id": "frame_123"
        }

        Args:
            data: JSON string from WebSocket
            stream_id: Stream identifier

        Returns:
            Parsed frame dict or None if invalid
        """
        try:
            frame = json.loads(data)

            # Validate required fields
            if frame.get("type") != "audio_chunk":
                logger.warning(f"Stream {stream_id}: unexpected frame type")
                return None

            if "audio_base64" not in frame:
                logger.warning(f"Stream {stream_id}: missing audio data")
                return None

            # Decode audio
            try:
                audio_data = base64.b64decode(frame["audio_base64"])
            except Exception as e:
                logger.error(f"Stream {stream_id}: base64 decode failed - {e}")
                return None

            # Return validated frame
            return {
                "type": frame.get("type"),
                "timestamp": frame.get("timestamp"),
                "audio_data": audio_data,
                "sample_rate": frame.get("sample_rate", 16000),
                "frame_id": frame.get("frame_id"),
                "stream_id": stream_id,
            }

        except Exception as e:
            logger.error(f"Stream {stream_id}: frame parse error - {e}")
            return None

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send error message to client.

        Args:
            websocket: WebSocket connection
            message: Error message
        """
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": message,
            }))
        except Exception as e:
            logger.error(f"Failed to send error to client: {e}")

    def get_stream_stats(self, stream_id: str) -> Optional[dict]:
        """Get statistics for a stream.

        Args:
            stream_id: Stream identifier

        Returns:
            Stats dict or None if stream not found
        """
        if stream_id not in self.active_streams:
            return None

        stream = self.active_streams[stream_id]
        return {
            "stream_id": stream_id,
            "frame_count": stream["frame_count"],
            "buffer_size": len(stream["buffer"]),
            "sample_rate": stream.get("sample_rate"),
        }

    def get_active_streams(self) -> list:
        """Get list of active stream IDs."""
        return list(self.active_streams.keys())

    async def close_stream(self, stream_id: str) -> None:
        """Close a stream connection.

        Args:
            stream_id: Stream identifier
        """
        if stream_id in self.active_streams:
            try:
                websocket = self.active_streams[stream_id]["websocket"]
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing stream {stream_id}: {e}")
