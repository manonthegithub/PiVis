import numpy as np
import pytest
from pivis.vision.detection import BoundingBox
from pivis.vision.loop import encode_jpeg


def _blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_encode_jpeg_no_boxes():
    frame = _blank_frame()
    result = encode_jpeg(frame, [])
    assert isinstance(result, bytes)
    assert result[:2] == b"\xff\xd8"  # JPEG magic


def test_encode_jpeg_with_boxes():
    frame = _blank_frame()
    boxes = [BoundingBox(0.1, 0.1, 0.5, 0.5, 0.9)]
    result = encode_jpeg(frame, boxes)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_encode_jpeg_does_not_mutate_frame():
    frame = _blank_frame()
    original = frame.copy()
    encode_jpeg(frame, [BoundingBox(0, 0, 1, 1, 0.9)])
    assert np.array_equal(frame, original)


def test_encode_jpeg_clips_boxes_to_frame():
    frame = _blank_frame()
    # Box extends beyond frame bounds — should not crash
    boxes = [BoundingBox(-0.1, -0.1, 1.5, 1.5, 0.9)]
    result = encode_jpeg(frame, boxes)
    assert result[:2] == b"\xff\xd8"
