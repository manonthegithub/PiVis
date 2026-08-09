import asyncio
import numpy as np
import pytest
from pivis.vision.detection import BoundingBox, DetectionResult
from pivis.state import AppState, DetectionEvent, Queues


def test_detection_event_fields():
    ev = DetectionEvent(
        boxes=[{"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5, "confidence": 0.9}],
        has_person=True,
        sensor_timestamp_ns=123456789,
    )
    assert ev.has_person is True
    assert ev.sensor_timestamp_ns == 123456789
    assert ev.boxes[0]["confidence"] == 0.9


def test_detection_result_carries_timestamp():
    result = DetectionResult(
        has_person=True,
        boxes=[BoundingBox(0.1, 0.1, 0.5, 0.5, 0.9)],
        sensor_timestamp_ns=987654321,
    )
    assert result.sensor_timestamp_ns == 987654321


def test_detection_result_no_person_carries_timestamp():
    result = DetectionResult(has_person=False, sensor_timestamp_ns=111)
    assert result.sensor_timestamp_ns == 111


def test_queues_has_detection_and_nal_and_fmp4():
    q = Queues()
    assert hasattr(q, 'detections')
    assert hasattr(q, 'nal_queue')
    assert hasattr(q, 'fmp4_queue')
