# birdcrop/utils.py
"""Utility functions for the birdcrop library."""

from pathlib import Path
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def find_image_files(input_paths: List[str], recursive: bool) -> List[Path]:
    """
    Finds all image files in the given paths (files or directories).

    Args:
        input_paths (List[str]): List of input file or directory paths.
        recursive (bool): Whether to search directories recursively.

    Returns:
        List[Path]: A sorted list of unique Path objects for found image files.
    """
    image_files: Set[Path] = set() # Use a set for automatic deduplication
    for path_str in input_paths:
        try:
            # Resolve paths early for consistency and absolute paths
            path = Path(path_str).resolve()
        except Exception as e:
             logger.warning(f"Could not resolve path '{path_str}', skipping: {e}")
             continue

        if not path.exists():
             logger.warning(f"Input path does not exist, skipping: {path}")
             continue

        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_files.add(path)
            else:
                logger.warning(f"Skipping non-image or unsupported file: {path}")
        elif path.is_dir():
            logger.info(f"Scanning directory: {path}{' (recursively)' if recursive else ''}")
            pattern = '**/*' if recursive else '*'
            try:
                for item in path.glob(pattern):
                    # Check again if it's a file and supported type after globbing
                    if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                        image_files.add(item)
            except Exception as e:
                 logger.error(f"Error scanning directory {path}: {e}")
        else:
            # This case might be less likely if exists() check passed, but good to have
            logger.warning(f"Input path is not a file or directory, skipping: {path}")

    if not image_files:
        logger.warning("No processable image files found in the specified inputs.")
        return []

    # Convert set to list and sort for deterministic order
    sorted_files = sorted(list(image_files))
    logger.info(f"Found {len(sorted_files)} unique image file(s) to process.")
    return sorted_files

