# birdcrop/cropper.py
"""Contains the core BirdCropper class for detection and cropping."""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
import os # Needed for os.stat
from typing import List, Dict, Any, Optional # Added Optional

# Import the new utility functions
from .utils import generate_output_path, get_exif_data

# Optional: Import custom exceptions if using them
# from .exceptions import ModelLoadError, ImageProcessingError, PredictionError, FileWriteError, DirectoryCreationError

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class BirdCropper:
    """
    Detects and crops birds from images using a YOLO model.
    Handles model loading, detection, sorting, cropping with margin,
    and flexible output path generation.
    """
    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 process_single: bool = True,
                 sort_by: str = "size",
                 class_id: int = 14,
                 margin: int = 0): # Added margin
        """
        Initializes the BirdCropper.

        Args:
            model_path (str): Path to the YOLOv8 model file.
            process_single (bool): If True, only process the single best bird per image.
            sort_by (str): Criterion ("confidence" or "size") to determine the best bird.
            class_id (int): The class ID to target for detection.
            margin (int): Pixel margin to add around the bounding box before cropping.

        Raises:
            ValueError: If sort_by is invalid or margin is negative.
            ModelLoadError: If the YOLO model cannot be loaded.
        """
        self.model_path = model_path
        self.process_single = process_single
        self.sort_by = sort_by
        self.class_id = class_id
        self.margin = margin # Store margin

        if sort_by not in ["confidence", "size"]:
             raise ValueError("sort_by must be 'confidence' or 'size'")
        if margin < 0:
             raise ValueError("margin cannot be negative")

        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.exception(f"Failed to load YOLO model from {self.model_path}: {e}")
            # raise ModelLoadError(...)
            raise

    def _calculate_area(self, box: np.ndarray) -> float:
        """Calculates the area of a bounding box."""
        x1, y1, x2, y2 = box
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return float(width * height)

    def _sort_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sorts detections based on the configured strategy."""
        if not detections:
            return []

        # Ensure size is calculated if sorting by size
        if self.sort_by == "size":
            for det in detections:
                if 'size' not in det:
                    det['size'] = self._calculate_area(det['box'])

        key_func = lambda x: x[self.sort_by] # Use 'conf' or 'size' directly
        detections.sort(key=key_func, reverse=True)
        logger.debug(f"Sorted {len(detections)} detections by {self.sort_by}.")
        return detections

    def detect_and_crop(self,
                        img_path: Path,
                        conf: float,
                        output_template: str, # New: template string
                        force_overwrite: bool # New: force flag
                       ) -> List[Path]:
        """
        Detects birds in a single image, crops them with margin, and saves them
        using a path template.

        Args:
            img_path (Path): Path to the input image file.
            conf (float): Confidence threshold for detection.
            output_template (str): The template string for generating output paths.
            force_overwrite (bool): If True, overwrite existing output files.

        Returns:
            list[Path]: A list of paths to the saved crop files.
        """
        # --- Pre-checks ---
        if not img_path.is_file():
            logger.warning(f"Input path is not a file, skipping: {img_path}")
            return []

        # --- Gather Static Input Data for Templating ---
        try:
            stat_info = img_path.stat() # Get file stats
        except Exception as e:
            logger.warning(f"Could not get file stats for {img_path}: {e}")
            stat_info = None # Handle missing stat info later

        exif_info = get_exif_data(img_path) # Get EXIF data

        # --- Image Reading ---
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Could not read image: {img_path}")
                return []
            img_height, img_width = img.shape[:2]
            logger.debug(f"Processing image: {img_path} ({img_width}x{img_height})")
        except Exception as e:
            logger.error(f"Error reading or processing image {img_path}: {e}")
            return []

        # --- Prediction ---
        try:
            results = self.model.predict(source=img, conf=conf, verbose=False)
        except Exception as e:
            logger.error(f"Error during YOLO prediction for {img_path}: {e}")
            return []

        # --- Detection Extraction ---
        all_detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() # Original boxes
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            for box, cls, conf_score in zip(boxes, classes, confidences):
                 if int(cls) == self.class_id:
                    detection_info = {
                        'box': box, # Original box
                        'cls': int(cls),
                        'conf': float(conf_score),
                        'size': self._calculate_area(box) # Area of original box
                    }
                    all_detections.append(detection_info)

        if not all_detections:
            logger.info(f"No birds (class {self.class_id}) detected in {img_path.name} with confidence >= {conf}.")
            return []

        # --- Sorting ---
        if len(all_detections) > 1 or self.process_single:
            all_detections = self._sort_detections(all_detections)

        # --- Determine which detections to process ---
        detections_to_process = all_detections[0:1] if self.process_single else all_detections
        logger.debug(f"Selected {len(detections_to_process)} detections to crop for {img_path.name}.")

        # --- Cropping and Saving ---
        saved_crops = []
        for i, det in enumerate(detections_to_process):
            nr = i + 1 # 1-based index for template {nr}
            box = det['box']

            # --- Calculate Coordinates with Margin ---
            x1_orig, y1_orig, x2_orig, y2_orig = map(int, box)

            # Apply margin
            x1 = x1_orig - self.margin
            y1 = y1_orig - self.margin
            x2 = x2_orig + self.margin
            y2 = y2_orig + self.margin

            # Clamp coordinates to image bounds AFTER applying margin
            x1_crop = max(0, x1)
            y1_crop = max(0, y1)
            x2_crop = min(img_width, x2)
            y2_crop = min(img_height, y2)

            # Check for valid crop dimensions AFTER clamping
            if x1_crop >= x2_crop or y1_crop >= y2_crop:
                logger.warning(f"Invalid crop dimensions after margin/clamping [x1={x1_crop}, y1={y1_crop}, x2={x2_crop}, y2={y2_crop}] for detection {nr} in {img_path.name}, skipping.")
                continue

            try:
                # --- Perform Cropping ---
                crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
                if crop.size == 0:
                    logger.warning(f"Calculated crop resulted in empty image for detection {nr} in {img_path.name}, skipping.")
                    continue

                crop_height, crop_width = crop.shape[:2]

                # --- Prepare Data for Templating ---
                template_data = {
                    'p': img_path,
                    'stat': stat_info, # The actual stat object (or None)
                    'exif': exif_info, # The EXIF dict (or None)
                    # Original detection data
                    'box': det['box'],
                    'cls': det['cls'],
                    'conf': det['conf'],
                    'size': det['size'],
                    'x1': x1_orig, # Original coords
                    'y1': y1_orig,
                    'x2': x2_orig,
                    'y2': y2_orig,
                    # Crop-specific data
                    'nr': nr,
                    'width': crop_width, # Actual crop dimensions
                    'height': crop_height,
                    'margin': self.margin,
                    'x1_crop': x1_crop, # Coords used for cropping
                    'y1_crop': y1_crop,
                    'x2_crop': x2_crop,
                    'y2_crop': y2_crop,
                }
                # Add specific stat/exif fields if they exist, handling None
                if stat_info:
                    template_data['st_size'] = stat_info.st_size
                    template_data['st_mtime'] = stat_info.st_mtime
                    # Add more stat fields as needed
                if exif_info:
                    # Add common EXIF fields, provide default if missing
                    template_data['DateTimeOriginal'] = exif_info.get('DateTimeOriginal', '')
                    template_data['Make'] = exif_info.get('Make', '')
                    template_data['Model'] = exif_info.get('Model', '')
                    # Add more EXIF fields as needed

                # --- Generate Output Path ---
                # The utility function now handles resolving and creating parent dirs
                output_path = generate_output_path(output_template, template_data, img_path)

                # --- Handle Overwrite/Collision ---
                write_allowed = False
                if force_overwrite:
                    if output_path.exists():
                         logger.debug(f"Overwriting existing file due to --force: {output_path}")
                    write_allowed = True
                elif not output_path.exists():
                    write_allowed = True
                else:
                    # Collision handling (only if force_overwrite is False)
                    logger.warning(f"Output file exists: {output_path}. Collision handling not implemented with templating (use --force or unique templates). Skipping.")
                    # TODO: Implement robust collision handling with templating if needed (e.g., append _N before suffix)
                    # For now, we just skip if --force is not used and file exists.
                    write_allowed = False


                # --- Save the crop ---
                if write_allowed:
                    success = cv2.imwrite(str(output_path), crop)
                    if success:
                        logger.info(f"Saved crop: {output_path} (Confidence: {det['conf']:.2f}, Size: {det['size']:.0f}px)")
                        saved_crops.append(output_path)
                    else:
                        logger.error(f"Failed to write crop file: {output_path}")
                        # Optional: raise FileWriteError(...)
                # else: # Already logged the reason for not writing

            except Exception as e:
                logger.exception(f"Error during cropping or saving for {img_path.name} (detection {nr}): {e}")

        return saved_crops

