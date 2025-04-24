# birdcrop/utils.py
"""Utility functions for the birdcrop library."""

import os
import piexif # Import piexif
import piexif.helper # For user comments
from pathlib import Path
import logging
from typing import List, Set, Dict, Any, Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'} # Keep this

# --- New Function: Safe EXIF Reading ---
def get_exif_data(image_path: Path) -> Optional[Dict[str, Any]]:
    """
    Attempts to read EXIF data from an image file.

    Handles potential errors gracefully. Currently focuses on JPEG/TIFF.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing EXIF data if successful,
                                  otherwise None. Returns simplified keys.
    """
    if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.tif', '.tiff']:
        logger.debug(f"EXIF reading skipped for non-JPEG/TIFF file: {image_path.name}")
        return None

    exif_data = {}
    try:
        exif_dict = piexif.load(str(image_path))
        # Simplify the structure and decode UserComment if present
        for ifd_name in exif_dict:
            if ifd_name == "thumbnail": # Skip thumbnail binary data
                continue
            for tag, value in exif_dict[ifd_name].items():
                tag_name = piexif.TAGS[ifd_name].get(tag, {}).get("name", f"{ifd_name}_{tag}") # Get readable tag name

                # Decode bytes to string where appropriate
                if isinstance(value, bytes):
                    # Special handling for UserComment (often needs specific decoding)
                    if tag_name == "UserComment":
                        try:
                             # piexif.helper handles common encodings like ASCII, JIS, Unicode
                            exif_data[tag_name] = piexif.helper.UserComment.load(value)
                        except Exception:
                             logger.debug(f"Could not decode UserComment for {image_path.name}")
                             # Fallback to representing as bytes or skipping
                             # exif_data[tag_name] = value
                             pass # Skip if undecodable
                    else:
                        try:
                            exif_data[tag_name] = value.decode('utf-8', errors='replace').strip('\x00')
                        except UnicodeDecodeError:
                            logger.debug(f"Could not decode tag '{tag_name}' as UTF-8 for {image_path.name}, storing raw bytes representation.")
                            # Store a representation instead of raw bytes
                            exif_data[tag_name] = f"bytes[{len(value)}]"
                else:
                    exif_data[tag_name] = value

        logger.debug(f"Successfully read EXIF data for {image_path.name}")
        return exif_data

    except FileNotFoundError:
        logger.warning(f"File not found during EXIF read: {image_path}")
        return None
    except piexif.InvalidImageDataError:
        logger.debug(f"No valid EXIF data found in {image_path.name}")
        return None
    except Exception as e:
        logger.warning(f"Error reading EXIF data for {image_path.name}: {e}", exc_info=False) # Keep log clean unless debugging
        return None


# --- New Class: SafeFormatter ---
class SafeFormatter(dict):
    """
    Dictionary subclass for str.format_map that ignores missing keys.
    """
    def __missing__(self, key):
        logger.warning(f"Template key '{key}' not found in provided data. Replacing with empty string.")
        return "" # Return empty string for missing keys

# --- New Function: Generate Output Path ---
def generate_output_path(
    template: str,
    data: Dict[str, Any],
    input_path: Path,
    default_suffix: str = ".jpg"
) -> Path:
    """
    Generates an output path based on a template string and data dictionary.

    Args:
        template (str): The format string template.
        data (Dict[str, Any]): Dictionary containing values for substitution.
        input_path (Path): The original input path, used as anchor for relative paths.
        default_suffix (str): The suffix to append if the template doesn't specify one.

    Returns:
        Path: The generated output Path object.
    """
    try:
        # Use SafeFormatter to avoid KeyError on missing keys
        formatted_str = template.format_map(SafeFormatter(data))
    except Exception as e:
        logger.error(f"Error formatting output path template '{template}': {e}. Using default naming.", exc_info=True)
        # Fallback to a simple default name in case of formatting errors
        nr = data.get('nr', 0)
        formatted_str = f"{input_path.stem}_crop_{nr}_fmt_error"

    # Create a preliminary Path object
    output_path = Path(formatted_str)

    # If the formatted path is relative, anchor it to the input path's parent
    if not output_path.is_absolute():
        output_path = input_path.parent / output_path

    # Ensure a suffix exists, applying default if needed
    if not output_path.suffix:
        output_path = output_path.with_suffix(default_suffix)

    # Resolve the path to clean it up (e.g., remove ../) and make it absolute
    try:
        resolved_path = output_path.resolve()
    except Exception as e: # Catch potential resolution errors (e.g., invalid chars)
        logger.error(f"Could not resolve generated path '{output_path}': {e}. Using fallback name in input directory.")
        nr = data.get('nr', 0)
        resolved_path = (input_path.parent / f"{input_path.stem}_crop_{nr}_resolve_error").with_suffix(default_suffix)


    # Ensure the parent directory for the final path exists *before* returning
    # This is crucial as the path might be anywhere now
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create parent directory {resolved_path.parent} for output file: {e}")
        # Decide how to handle this: maybe return a path in a known-good location?
        # For now, let's return the problematic path and let the writing fail later.
        pass # Or raise DirectoryCreationError(..)

    return resolved_path


# --- Keep find_image_files (no changes needed here) ---
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
