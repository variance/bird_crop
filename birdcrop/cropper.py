# birdcrop/cropper.py
"""Contains the core BirdCropper class for detection and cropping."""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
import os
from typing import List, Dict, Any, Optional, Set, Union # Added Set, Union

# Import utility functions
from .utils import generate_output_path, get_exif_data

# Optional: Import custom exceptions
# from .exceptions import ModelLoadError, ...

logger = logging.getLogger(__name__)

class BirdCropper:
    """
    Detects and crops objects from images using a YOLO model.
    Handles model loading, targeting specific classes (by name or ID),
    sorting, cropping with margin, and flexible output path generation.
    """
    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 target_classes: List[Union[str, int]] = None, # New: List of names/IDs
                 process_single: bool = True,
                 sort_by: str = "size",
                 margin: int = 0):
        """
        Initializes the BirdCropper.

        Args:
            model_path (str): Path to the YOLOv8 model file.
            target_classes (List[Union[str, int]]): List of class names (str) or
                                                    class IDs (int) to detect.
                                                    Defaults to ['bird'] if None.
            process_single (bool): If True, only process the single best object per image
                                   across all target classes.
            sort_by (str): Criterion ("confidence" or "size") to determine the best object.
            margin (int): Pixel margin to add around the bounding box before cropping.

        Raises:
            ValueError: If sort_by is invalid, margin is negative, or no valid target
                        classes could be resolved.
            FileNotFoundError: If the model_path does not exist.
            Exception: Propagates exceptions from YOLO model loading.
        """
        self.model_path = model_path
        self.process_single = process_single
        self.sort_by = sort_by
        self.margin = margin

        if target_classes is None:
            target_classes = ['bird'] # Default to bird if nothing specified

        if sort_by not in ["confidence", "size"]:
             raise ValueError("sort_by must be 'confidence' or 'size'")
        if margin < 0:
             raise ValueError("margin cannot be negative")
        if not isinstance(target_classes, list) or not target_classes:
             raise ValueError("target_classes must be a non-empty list of strings or integers")

        # --- Model Loading and Class Name Extraction ---
        model_file = Path(model_path)
        if not model_file.is_file():
             raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully from {self.model_path}")
            # Extract class names (ID -> name mapping)
            self.class_id_to_name: Dict[int, str] = self.model.names
            logger.debug(f"Model class names loaded: {self.class_id_to_name}")
        except Exception as e:
            logger.exception(f"Failed to load YOLO model from {self.model_path}: {e}")
            # Consider raising a specific ModelLoadError here
            raise # Re-raise original exception

        # --- Resolve Target Class IDs ---
        self.target_class_ids: Set[int] = set()
        name_to_id: Dict[str, int] = {name.lower(): id for id, name in self.class_id_to_name.items()} # For case-insensitive name lookup

        for target in target_classes:
            if isinstance(target, int):
                if target in self.class_id_to_name:
                    self.target_class_ids.add(target)
                else:
                    logger.warning(f"Requested class ID {target} is not present in the loaded model's classes. Ignoring.")
            elif isinstance(target, str):
                target_lower = target.lower()
                if target_lower in name_to_id:
                    self.target_class_ids.add(name_to_id[target_lower])
                else:
                    logger.warning(f"Requested class name '{target}' not found in the loaded model's classes. Ignoring.")
            else:
                logger.warning(f"Invalid item '{target}' in target_classes list. Must be int or str. Ignoring.")

        if not self.target_class_ids:
            raise ValueError("No valid target classes could be resolved from the input list for the loaded model.")

    def _calculate_area(self, box: np.ndarray) -> float:
        x1, y1, x2, y2 = box
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return float(width * height)

    def _sort_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []
        if self.sort_by == "size":
            for det in detections:
                if 'size' not in det:
                    det['size'] = self._calculate_area(det['box'])
        key_func = lambda x: x[self.sort_by]
        detections.sort(key=key_func, reverse=True)
        logger.debug(f"Sorted {len(detections)} detections by {self.sort_by}.")
        return detections

    def detect_and_crop(self,
                        img_path: Path,
                        conf: float,
                        output_template: str,
                        force_overwrite: bool
                       ) -> List[Path]:
        """
        Detects target objects in an image, crops them with margin, and saves them
        using a path template.

        Args:
            img_path (Path): Path to the input image file.
            conf (float): Confidence threshold for detection.
            output_template (str): The template string for generating output paths.
            force_overwrite (bool): If True, overwrite existing output files.

        Returns:
            list[Path]: A list of paths to the saved crop files.
        """
        # --- Pre-checks, Stat/EXIF gathering, Image Reading ---
        if not img_path.is_file(): logger.warning(f"Input path is not a file, skipping: {img_path}"); return []
        try: stat_info = img_path.stat()
        except Exception as e: logger.warning(f"Could not get file stats for {img_path}: {e}"); stat_info = None
        exif_info = get_exif_data(img_path)
        try:
            img = cv2.imread(str(img_path))
            if img is None: logger.error(f"Could not read image: {img_path}"); return []
            img_height, img_width = img.shape[:2]
            logger.debug(f"Processing image: {img_path} ({img_width}x{img_height})")
        except Exception as e: logger.error(f"Error reading or processing image {img_path}: {e}"); return []

        # --- Prediction ---
        try: results = self.model.predict(source=img, conf=conf, verbose=False)
        except Exception as e: logger.error(f"Error during YOLO prediction for {img_path}: {e}"); return []

        # --- Detection Extraction ---
        all_detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            for box, cls, conf_score in zip(boxes, classes, confidences):
                detected_cls_id = int(cls)
                # --- Check against the set of target IDs ---
                if detected_cls_id in self.target_class_ids:
                    detection_info = {
                        'box': box,
                        'cls': detected_cls_id, # Keep the ID
                        'conf': float(conf_score),
                        'size': self._calculate_area(box)
                    }
                    all_detections.append(detection_info)

        if not all_detections:
            logger.info(f"No target objects detected in {img_path.name} with confidence >= {conf}.")
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
            nr = i + 1
            box = det['box']
            detected_cls_id = det['cls'] # Get the class ID for this detection

            # --- Calculate Coordinates with Margin ---
            x1_orig, y1_orig, x2_orig, y2_orig = map(int, box)
            x1, y1 = x1_orig - self.margin, y1_orig - self.margin
            x2, y2 = x2_orig + self.margin, y2_orig + self.margin
            x1_crop, y1_crop = max(0, x1), max(0, y1)
            x2_crop, y2_crop = min(img_width, x2), min(img_height, y2)
            if x1_crop >= x2_crop or y1_crop >= y2_crop:
                logger.warning(f"Invalid crop dimensions after margin/clamping for detection {nr} in {img_path.name}, skipping.")
                continue

            try:
                # --- Perform Cropping ---
                crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
                if crop.size == 0: logger.warning(f"Empty crop for detection {nr} in {img_path.name}, skipping."); continue
                crop_height, crop_width = crop.shape[:2]

                # --- Prepare Data for Templating ---
                # --- Add the 'category' field ---
                category_name = self.class_id_to_name.get(detected_cls_id, f"unknown_{detected_cls_id}")

                template_data = {
                    'p': img_path,
                    'stat': stat_info,
                    'exif': exif_info,
                    # Original detection data
                    'box': det['box'],
                    'cls': detected_cls_id, # Class ID
                    'category': category_name, # Class Name <<< NEW
                    'conf': det['conf'],
                    'size': det['size'],
                    'x1': x1_orig, 'y1': y1_orig, 'x2': x2_orig, 'y2': y2_orig,
                    # Crop-specific data
                    'nr': nr,
                    'width': crop_width, 'height': crop_height,
                    'margin': self.margin,
                    'x1_crop': x1_crop, 'y1_crop': y1_crop, 'x2_crop': x2_crop, 'y2_crop': y2_crop,
                }
                # Add specific stat/exif fields (no changes needed here)
                if stat_info:
                    template_data['st_size'] = stat_info.st_size
                    template_data['st_mtime'] = stat_info.st_mtime
                if exif_info:
                    template_data['DateTimeOriginal'] = exif_info.get('DateTimeOriginal', '')
                    template_data['Make'] = exif_info.get('Make', '')
                    template_data['Model'] = exif_info.get('Model', '')

                # --- Generate Output Path ---
                output_path = generate_output_path(output_template, template_data, img_path)

                # --- Handle Overwrite/Collision ---
                write_allowed = False
                if force_overwrite:
                    if output_path.exists(): logger.debug(f"Overwriting existing file due to --force: {output_path}")
                    write_allowed = True
                elif not output_path.exists():
                    write_allowed = True
                else:
                    logger.warning(f"Output file exists: {output_path}. Skipping (use --force to overwrite).")
                    write_allowed = False

                # --- Save the crop ---
                if write_allowed:
                    success = cv2.imwrite(str(output_path), crop)
                    if success:
                        logger.info(f"Saved crop: {output_path} (Class: {category_name}, Conf: {det['conf']:.2f}, Size: {det['size']:.0f}px)")
                        saved_crops.append(output_path)
                    else:
                        logger.error(f"Failed to write crop file: {output_path}")

            except Exception as e:
                logger.exception(f"Error during cropping or saving for {img_path.name} (detection {nr}): {e}")

        return saved_crops

