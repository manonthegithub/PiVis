import numpy as np
import pytest
from pivis.vision.detection import BoundingBox
from pivis.vision.loop import encode_jpeg


def _blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_encode_jpeg_no_boxes():
    ann, side = encode_jpeg(_blank_frame(), [])
    assert ann[:2] == b"\xff\xd8"
    assert side[:2] == b"\xff\xd8"


def test_encode_jpeg_with_boxes():
    boxes = [BoundingBox(0.1, 0.1, 0.5, 0.5, 0.9)]
    ann, side = encode_jpeg(_blank_frame(), boxes)
    assert len(ann) > 0
    assert len(side) > 0


def test_encode_jpeg_does_not_mutate_frame():
    frame = _blank_frame()
    original = frame.copy()
    encode_jpeg(frame, [BoundingBox(0, 0, 1, 1, 0.9)])
    assert np.array_equal(frame, original)


def test_encode_jpeg_clips_boxes_to_frame():
    # Box extends beyond frame bounds — should not crash
    ann, _ = encode_jpeg(_blank_frame(), [BoundingBox(-0.1, -0.1, 1.5, 1.5, 0.9)])
    assert ann[:2] == b"\xff\xd8"


def test_encode_jpeg_side_is_wider():
    ann, side = encode_jpeg(_blank_frame(480, 640), [])
    import cv2
    ann_w = cv2.imdecode(np.frombuffer(ann, np.uint8), cv2.IMREAD_COLOR).shape[1]
    side_w = cv2.imdecode(np.frombuffer(side, np.uint8), cv2.IMREAD_COLOR).shape[1]
    assert side_w == ann_w * 2
