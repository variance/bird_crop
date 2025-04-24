# birdcrop/exceptions.py
"""Custom exceptions for the birdcrop library."""

class BirdCropError(Exception):
    """Base exception for the birdcrop library."""
    pass

class ModelLoadError(BirdCropError):
    """Raised when the YOLO model fails to load."""
    pass

class ImageProcessingError(BirdCropError):
    """Raised during image reading or initial processing."""
    pass

class PredictionError(BirdCropError):
    """Raised during the YOLO prediction phase."""
    pass

class FileWriteError(BirdCropError):
    """Raised when saving a cropped image fails."""
    pass

class DirectoryCreationError(BirdCropError):
    """Raised when creating an output directory fails."""
    pass
