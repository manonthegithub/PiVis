import numpy as np
import pytest
from pivis.vision.detection import BoundingBox, DetectionEngine, DetectionResult, _iou


def _make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_engine_with_mock_session(predictions: np.ndarray, threshold=0.5):
    """Inject a fake ONNX session so tests don't need the model file."""
    engine = DetectionEngine(confidence_threshold=threshold)

    class FakeSession:
        def get_inputs(self):
            class I:
                name = "images"
            return [I()]

        def run(self, _, __):
            return [predictions]

    engine._session = FakeSession()
    engine._input_name = "images"
    return engine


def _pred(cx, cy, w, h, person_conf, other_conf=0.0):
    """Build one YOLOv8 prediction row: [cx,cy,w,h, class0_score, class1..79]."""
    row = np.zeros(84, dtype=np.float32)
    row[:4] = [cx, cy, w, h]
    row[4] = person_conf   # class 0 = person
    row[5] = other_conf
    return row


def test_detect_no_person():
    preds = np.zeros((1, 84, 8400), dtype=np.float32)
    engine = _make_engine_with_mock_session(preds)
    result = engine.detect(_make_frame(), sensor_timestamp_ns=0)
    assert result.has_person is False
    assert result.boxes == []


def test_detect_person():
    row = _pred(cx=320, cy=240, w=100, h=200, person_conf=0.9)
    preds = np.zeros((1, 84, 8400), dtype=np.float32)
    preds[0, :, 0] = row
    engine = _make_engine_with_mock_session(preds)
    result = engine.detect(_make_frame(), sensor_timestamp_ns=123)
    assert result.has_person is True
    assert len(result.boxes) == 1
    assert result.confidence == pytest.approx(0.9, abs=1e-4)
    assert result.sensor_timestamp_ns == 123


def test_detect_timestamp_propagated():
    preds = np.zeros((1, 84, 8400), dtype=np.float32)
    engine = _make_engine_with_mock_session(preds)
    result = engine.detect(_make_frame(), sensor_timestamp_ns=999_000_000)
    assert result.sensor_timestamp_ns == 999_000_000


def test_detect_below_threshold():
    row = _pred(cx=320, cy=240, w=100, h=200, person_conf=0.3)
    preds = np.zeros((1, 84, 8400), dtype=np.float32)
    preds[0, :, 0] = row
    engine = _make_engine_with_mock_session(preds, threshold=0.5)
    result = engine.detect(_make_frame())
    assert result.has_person is False


def test_nms_removes_overlapping():
    boxes = [
        BoundingBox(0.1, 0.1, 0.5, 0.5, confidence=0.9),
        BoundingBox(0.11, 0.11, 0.51, 0.51, confidence=0.8),  # heavily overlaps
        BoundingBox(0.6, 0.6, 0.9, 0.9, confidence=0.7),      # separate
    ]
    kept = DetectionEngine._nms(boxes)
    assert len(kept) == 2
    assert kept[0].confidence == pytest.approx(0.9)
    assert kept[1].confidence == pytest.approx(0.7)


def test_iou_identical():
    b = BoundingBox(0, 0, 1, 1, 1.0)
    assert _iou(b, b) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = BoundingBox(0, 0, 0.4, 0.4, 1.0)
    b = BoundingBox(0.6, 0.6, 1.0, 1.0, 1.0)
    assert _iou(a, b) == pytest.approx(0.0)


def test_detect_raises_without_load():
    engine = DetectionEngine()
    with pytest.raises(RuntimeError, match="load()"):
        engine.detect(_make_frame())


def test_detect_raises_wrong_frame_shape():
    engine = _make_engine_with_mock_session(np.zeros((1, 84, 8400)))
    with pytest.raises(ValueError):
        engine.detect(np.zeros((480, 640)))  # missing channel dim


def test_load_raises_missing_model(tmp_path):
    engine = DetectionEngine()
    with pytest.raises(FileNotFoundError):
        engine.load(tmp_path / "nonexistent.onnx")
