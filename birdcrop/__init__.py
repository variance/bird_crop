# birdcrop/__init__.py
"""
BirdCrop Library: Detect and crop birds from images using YOLO models.
"""

# Import key components to make them available directly from the package
from .cropper import BirdCropper
from .utils import find_image_files

# Optional: Import custom exceptions if defined
# from .exceptions import BirdCropError, ModelLoadError, ImageProcessingError, PredictionError, FileWriteError, DirectoryCreationError

# Define package version (optional but recommended)
__version__ = "0.1.2"
__date__ = "2025-06-05"

# Define what gets imported with 'from birdcrop import *' (optional)
__all__ = [
    'BirdCropper',
    'find_image_files',
    # Add exception names here if defined and desired
    '__version__',
    '__date__',
]

# You could potentially add a top-level convenience function here if needed,
# e.g., a function that takes inputs and options and runs the whole process.
