"""Detection backend adapters used by BirdCrop."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A backend-independent object detection."""

    box: np.ndarray
    class_id: int
    confidence: float


class DetectionBackend(Protocol):
    """Common contract implemented by all supported detection backends."""

    class_names: Dict[int, str]

    def predict(
        self,
        image: np.ndarray,
        confidence: float,
        tiling: bool = False,
        overlap_ratio: float = 0.25,
    ) -> Iterable[Detection]:
        """Run detection and return boxes in original image coordinates."""


def _as_numpy(value: Any) -> np.ndarray:
    """Convert torch tensors and array-like backend values to NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize_class_names(names: Any) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    if isinstance(names, (list, tuple)):
        return {class_id: str(name) for class_id, name in enumerate(names)}
    raise ValueError("The selected model does not expose class names.")


def _extract_detections(results: Any) -> Iterable[Detection]:
    result_list = results if isinstance(results, (list, tuple)) else [results]
    for result in result_list:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        coordinates = _as_numpy(boxes.xyxy)
        class_ids = _as_numpy(boxes.cls)
        confidences = _as_numpy(boxes.conf)
        for box, class_id, confidence in zip(coordinates, class_ids, confidences):
            yield Detection(
                box=np.asarray(box, dtype=float),
                class_id=int(class_id),
                confidence=float(confidence),
            )


class UltralyticsBackend:
    """Adapter for the optional legacy Ultralytics backend."""

    _tiling_lock = threading.Lock()
    _tiling_warning_emitted = False

    def __init__(self, model_path: str):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.class_names = _normalize_class_names(self.model.names)
        self._is_yolo26 = "yolo26" in model_path.lower()

    def predict(
        self,
        image: np.ndarray,
        confidence: float,
        tiling: bool = False,
        overlap_ratio: float = 0.25,
    ) -> Iterable[Detection]:
        if tiling and not UltralyticsBackend._tiling_warning_emitted:
            # Ensure that only one thread emits the warning about tiling not being supported by the Ultralytics backend.
            with UltralyticsBackend._tiling_lock:
                if not UltralyticsBackend._tiling_warning_emitted:
                    logger.warning(
                        "The Ultralytics backend does not support native tiled inference; "
                        "falling back to standard prediction."
                    )
                UltralyticsBackend._tiling_warning_emitted = True
        results = self.model.predict(
            source=image,
            conf=confidence,
            verbose=False,
            nms=False if self._is_yolo26 else True,
        )
        return _extract_detections(results)


class LibreYOLOBackend:
    """Adapter for the MIT-licensed LibreYOLO backend."""

    def __init__(self, model_path: str):
        from libreyolo import LibreYOLO

        self.model = LibreYOLO(model_path)
        self.class_names = _normalize_class_names(self.model.names)

    def predict(
        self,
        image: np.ndarray,
        confidence: float,
        tiling: bool = False,
        overlap_ratio: float = 0.25,
    ) -> Iterable[Detection]:
        results = self.model.predict(
            source=image,
            conf=confidence,
            tiling=tiling,
            overlap_ratio=overlap_ratio,
        )
        return _extract_detections(results)


def create_backend(name: str, model_path: str) -> DetectionBackend:
    """Create the selected backend while keeping optional imports lazy."""
    if name == "ultralytics":
        return UltralyticsBackend(model_path)
    if name == "libreyolo":
        return LibreYOLOBackend(model_path)
    raise ValueError(f"Unsupported backend: {name}")


def list_model_classes(name: str, model_path: str) -> Dict[int, str]:
    """Load a model through the selected backend and return its classes."""
    backend = create_backend(name, model_path)
    return backend.class_names
