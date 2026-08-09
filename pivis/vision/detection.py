import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass
class DetectionResult:
    has_person: bool
    boxes: list[BoundingBox] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: float = 0.0
    sensor_timestamp_ns: int = 0


class DetectionEngine:
    def __init__(self, confidence_threshold: float = 0.5, input_size: int = 320) -> None:
        self._threshold = confidence_threshold
        self._input_size = input_size
        self._session = None

    def load(self, model_path: Path | str) -> None:
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4  # Pi5 has 4 cores
        self._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        logger.info("DetectionEngine loaded: %s", model_path)

    def detect(self, frame: np.ndarray, sensor_timestamp_ns: int = 0) -> DetectionResult:
        if self._session is None:
            raise RuntimeError("Call load() before detect()")

        t0 = time.monotonic()
        blob = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: blob})
        boxes = self._postprocess(outputs, frame.shape)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if elapsed_ms > 500:
            logger.warning("Slow inference: %.0fms", elapsed_ms)

        if not boxes:
            return DetectionResult(has_person=False, timestamp=time.time(), sensor_timestamp_ns=sensor_timestamp_ns)

        best = max(boxes, key=lambda b: b.confidence)
        return DetectionResult(
            has_person=True,
            boxes=boxes,
            confidence=best.confidence,
            timestamp=time.time(),
            sensor_timestamp_ns=sensor_timestamp_ns,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 frame, got {frame.shape}")

        resized = cv2.resize(frame, (self._input_size, self._input_size))
        blob = resized.astype(np.float32) / 255.0
        return blob.transpose(2, 0, 1)[np.newaxis]  # NCHW

    def _postprocess(
        self, outputs: list, original_shape: tuple
    ) -> list[BoundingBox]:
        # YOLOv8 output: [1, 84, 8400] — 4 box coords + 80 class scores
        preds = outputs[0][0].T  # (8400, 84)
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = (class_ids == _PERSON_CLASS_ID) & (confidences >= self._threshold)
        if not mask.any():
            return []

        p = preds[mask]
        c = confidences[mask]
        s = self._input_size
        x1 = np.clip((p[:, 0] - p[:, 2] / 2) / s, 0, 1)
        y1 = np.clip((p[:, 1] - p[:, 3] / 2) / s, 0, 1)
        x2 = np.clip((p[:, 0] + p[:, 2] / 2) / s, 0, 1)
        y2 = np.clip((p[:, 1] + p[:, 3] / 2) / s, 0, 1)

        boxes = [BoundingBox(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(c[i]))
                 for i in range(len(p))]
        return self._nms(boxes)

    @staticmethod
    def _nms(boxes: list[BoundingBox], iou_threshold: float = 0.45) -> list[BoundingBox]:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        kept = []
        for box in boxes:
            if all(_iou(box, k) < iou_threshold for k in kept):
                kept.append(box)
        return kept


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)
