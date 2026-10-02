# birdcrop/__init__.py
"""
BirdCrop Library: Detect and crop birds from images using YOLO models.
"""

# Import key components to make them available directly from the package
from .cropper import BirdCropper
from .utils import find_image_files
from .exceptions import (
    BirdCropError,
    DirectoryCreationError,
    FileWriteError,
    ImageProcessingError,
    ModelLoadError,
    PredictionError,
)

__version__ = "0.1.7"
__date__ = "2026-10-02"

__all__ = [
    'BirdCropper',
    'find_image_files',
    'BirdCropError',
    'DirectoryCreationError',
    'FileWriteError',
    'ImageProcessingError',
    'ModelLoadError',
    'PredictionError',
    '__version__',
    '__date__',
]
