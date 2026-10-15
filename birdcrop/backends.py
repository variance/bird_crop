"""Detection backend adapters used by BirdCrop."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path(user_cache_dir("birdcrop","variance")) / "models" # use my github username as the author name

# CAVEAT: The following URL may need to be updated for future releases of ultralytics/assets!
# The companion birdcrop-upgrade command can check for newer releases and download them.
_YOLO_RELEASE_URL_PREFIX = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"

# Model size mapping for YOLO26
YOLO26_MODEL_SIZES = {
    "nano":   ("yolo26n.pt", _YOLO_RELEASE_URL_PREFIX + "yolo26n.pt"),
    "small":  ("yolo26s.pt", _YOLO_RELEASE_URL_PREFIX + "yolo26s.pt"),
    "medium": ("yolo26m.pt", _YOLO_RELEASE_URL_PREFIX + "yolo26m.pt"),
    "large":  ("yolo26l.pt", _YOLO_RELEASE_URL_PREFIX + "yolo26l.pt"),
    "xlarge": ("yolo26x.pt", _YOLO_RELEASE_URL_PREFIX + "yolo26x.pt"),
}
    
YOLO9_MODEL_SIZES = {
    "tiny": ("LibreYOLO9t.pt", None),
    "small": ("LibreYOLO9s.pt", None),
    "medium": ("LibreYOLO9m.pt", None),
    "compact": ("LibreYOLO9c.pt", None),
}

@dataclass(frozen=True)
class ModelsInfo:
    model_sizes: Mapping[str, Tuple[str, Optional[str]]]
    default_model_size: str

MODELS_INFO_BY_BACKEND = {
    "yolo9": ModelsInfo(model_sizes=YOLO9_MODEL_SIZES, default_model_size="small"),
    "ultralytics": ModelsInfo(model_sizes=YOLO26_MODEL_SIZES, default_model_size="small"),
}
DEFAULT_BACKEND = "yolo9"  # Default backend if none specified

# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------

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

def _parse_version_parts(value: str) -> Tuple[int, ...]:
    """Extract numeric version parts from strings like '8.3.10' or '8.3.10rc1'."""
    parts = re.findall(r"\d+", value)
    if not parts:
        return (0,)
    return tuple(int(p) for p in parts[:4])


def _fetch_json(url: str, timeout: float = 2.0) -> Dict[str, Any] | None:
    """Fetch JSON from URL with a short timeout; return None on network/API errors."""
    req = Request(url, headers={"User-Agent": "birdcrop-update-check/1.0"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def check_ultralytics_model_release_update(current_assets_tag: str, model_filename: str | None = None, timeout: float = 2.0):
    """Check GitHub releases for newer ultralytics/assets model tags."""
    payload = _fetch_json("https://api.github.com/repos/ultralytics/assets/releases/latest", timeout=timeout)
    if not payload:
        logger.debug("Update check: could not query GitHub latest ultralytics/assets release.")
        return

    latest_tag = str(payload.get("tag_name", "")).strip()
    if not latest_tag:
        logger.debug("Update check: GitHub response did not include tag_name.")
        return

    if _parse_version_parts(latest_tag) <= _parse_version_parts(current_assets_tag):
        logger.info("Model assets tag is up to date (configured=%s, latest=%s).", current_assets_tag, latest_tag)
        return

    assets = payload.get("assets") or []
    if model_filename:
        has_matching_asset = any(a.get("name") == model_filename for a in assets)
        if has_matching_asset:
            logger.warning(
                "Newer model release available for %s: configured tag=%s, latest tag=%s.",
                model_filename,
                current_assets_tag,
                latest_tag,
            )
        else:
            logger.warning(
                "Newer ultralytics/assets release detected (configured tag=%s, latest tag=%s), "
                "but '%s' is not listed in latest assets.",
                current_assets_tag,
                latest_tag,
                model_filename,
            )
    else:
        logger.warning(
            "Newer ultralytics/assets release detected: configured tag=%s, latest tag=%s.",
            current_assets_tag,
            latest_tag,
        )


def check_ultralytics_update(timeout: float = 2.0):
    """Check PyPI for newer ultralytics package versions."""
    try:
        local_version = metadata.version("ultralytics")
    except metadata.PackageNotFoundError:
        logger.debug("Update check: ultralytics package not found in environment.")
        return
    except Exception as exc:
        logger.debug(f"Update check: could not read local ultralytics version: {exc}")
        return

    payload = _fetch_json("https://pypi.org/pypi/ultralytics/json", timeout=timeout)
    if not payload:
        logger.debug("Update check: could not query PyPI for ultralytics.")
        return

    latest_version = str(payload.get("info", {}).get("version", "")).strip()
    if not latest_version:
        logger.debug("Update check: PyPI response did not contain ultralytics version.")
        return

    if _parse_version_parts(latest_version) > _parse_version_parts(local_version):
        logger.warning(
            "ultralytics update available: local=%s latest=%s (PyPI)",
            local_version,
            latest_version,
        )
    else:
        logger.info("ultralytics is up to date (local=%s, latest=%s).", local_version, latest_version)

# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------

def create_backend(name: str, model_path: str) -> DetectionBackend:
    """Create the selected backend while keeping optional imports lazy."""
    if name == "ultralytics":
        return UltralyticsBackend(model_path)
    if name in ["yolo9"]:  # The LibreYOLO backend will be used for other model families in the future, but for now we only support yolo9.
        return LibreYOLOBackend(model_path)
    raise ValueError(f"Unsupported backend: {name}")


def list_model_classes(name: str, model_path: str) -> Dict[int, str]:
    """Load a model through the selected backend and return its classes."""
    backend = create_backend(name, model_path)
    return backend.class_names


def find_best_local_model(model_dir: Path = DEFAULT_MODEL_DIR, backend: str = DEFAULT_BACKEND) -> Tuple[str, str] | None:
    """
    Sucht im angegebenen Ordner nach dem besten vorhandenen YOLO-Modell.
    Bevorzugt größere/genauere Modelle, falls mehrere existieren.
    Rückgabe: Tuple (model_filename, model_size_key) oder None.
    CAVEAT: Die größten Modelle sind nicht immer die besten für alle Szenarien. User sollten die Leistung
    (Anzahl der erkannten Objekte, Genauigkeit, Rechenzeit) mit eigenen Testbildern validieren!
    """
    if backend not in MODELS_INFO_BY_BACKEND:
        raise ValueError(f"Unsupported backend: {backend}")
    if not model_dir.is_dir():
        return None
    model_sizes = MODELS_INFO_BY_BACKEND[backend].model_sizes

    # Präferenzreihenfolge für die automatische Erkennung vorhandener Modelle
    preference_order = ["xlarge", "large", "compact","medium", "small", "nano", "tiny"]
    
    for size in preference_order:
        if size not in model_sizes:
            continue
        filename, _ = model_sizes[size]
        if (model_dir / filename).is_file():
            return filename, size
            
    return None
