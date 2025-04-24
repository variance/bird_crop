# birdcrop/cropper.py
"""Contains the core BirdCropper class for detection and cropping."""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
import os
import json # Import json for metadata saving
from typing import List, Dict, Any, Optional, Set, Union, Tuple # Added Tuple
from collections import defaultdict

# Import utility functions
from .utils import generate_output_path, get_exif_data

logger = logging.getLogger(__name__)

# Helper to convert NumPy types for JSON serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path): # Handle Path objects
            return str(obj)
        if isinstance(obj, os.stat_result): # Handle stat results
             # Convert stat_result to a dictionary
             return {attr: getattr(obj, attr) for attr in dir(obj) if attr.startswith('st_')}
        return super(NpEncoder, self).default(obj)


class BirdCropper:
    """
    Detects and crops objects from images using a YOLO model.
    Handles model loading, targeting specific classes (by name or ID),
    sorting, cropping with margin, flexible output path generation,
    metadata saving, and dry run simulation.
    """
    def __init__(self, model_path: str = "yolov8n.pt", target_classes: List[Union[str, int]] = None, process_single: bool = True, sort_by: str = "size", margin: int = 0):
        self.model_path = model_path; self.process_single = process_single; self.sort_by = sort_by; self.margin = margin
        if target_classes is None: target_classes = ['bird']
        if sort_by not in ["confidence", "size"]: raise ValueError("sort_by must be 'confidence' or 'size'")
        if margin < 0: raise ValueError("margin cannot be negative")
        if not isinstance(target_classes, list) or not target_classes: raise ValueError("target_classes must be a non-empty list")
        model_file = Path(model_path);
        if not model_file.is_file(): raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            self.model = YOLO(self.model_path); logger.info(f"YOLO model loaded successfully from {self.model_path}")
            self.class_id_to_name: Dict[int, str] = self.model.names; logger.debug(f"Model class names loaded: {self.class_id_to_name}")
        except Exception as e: logger.exception(f"Failed to load YOLO model from {self.model_path}: {e}"); raise
        self.target_class_ids: Set[int] = set(); name_to_id: Dict[str, int] = {name.lower(): id for id, name in self.class_id_to_name.items()}
        for target in target_classes:
            if isinstance(target, int):
                if target in self.class_id_to_name: self.target_class_ids.add(target)
                else: logger.warning(f"Requested class ID {target} not in model. Ignoring.")
            elif isinstance(target, str):
                target_lower = target.lower()
                if target_lower in name_to_id: self.target_class_ids.add(name_to_id[target_lower])
                else: logger.warning(f"Requested class name '{target}' not found in model. Ignoring.")
            else: logger.warning(f"Invalid item '{target}' in target_classes list. Ignoring.")
        if not self.target_class_ids: raise ValueError("No valid target classes resolved for the loaded model.")

    def _calculate_area(self, box: np.ndarray) -> float:
        x1, y1, x2, y2 = box; width = max(0, x2 - x1); height = max(0, y2 - y1); return float(width * height)

    def _sort_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections: return []
        if self.sort_by == "size":
            for det in detections:
                if 'size' not in det: det['size'] = self._calculate_area(det['box'])
        key_func = lambda x: x[self.sort_by]; detections.sort(key=key_func, reverse=True)
        logger.debug(f"Sorted {len(detections)} detections by {self.sort_by}."); return detections

    def detect_and_crop(self,
                        img_path: Path,
                        conf: float,
                        output_template: str,
                        force_overwrite: bool,
                        dry_run: bool,          # New flag
                        save_metadata: bool     # New flag
                       ) -> Tuple[List[Path], List[Path]]: # Return tuple of saved paths
        """
        Detects target objects, crops them, and optionally saves metadata.
        Can simulate operations with dry_run.

        Args:
            img_path (Path): Path to the input image file.
            conf (float): Confidence threshold for detection.
            output_template (str): Template for generating output paths.
            force_overwrite (bool): If True, overwrite existing output files.
            dry_run (bool): If True, simulate actions without writing files.
            save_metadata (bool): If True, save metadata as JSON alongside crops.

        Returns:
            Tuple[List[Path], List[Path]]: A tuple containing:
                - List of paths to the saved/simulated crop files.
                - List of paths to the saved/simulated metadata files.
        """
        # --- Pre-checks, Stat/EXIF gathering, Image Reading ---
        if not img_path.is_file(): logger.warning(f"Input path is not a file, skipping: {img_path}"); return [], []
        try: stat_info = img_path.stat()
        except Exception as e: logger.warning(f"Could not get file stats for {img_path}: {e}"); stat_info = None
        exif_info = get_exif_data(img_path)
        try:
            img = cv2.imread(str(img_path))
            if img is None: logger.error(f"Could not read image: {img_path}"); return [], []
            img_height, img_width = img.shape[:2]
            logger.debug(f"Processing image: {img_path} ({img_width}x{img_height})")
        except Exception as e: logger.error(f"Error reading or processing image {img_path}: {e}"); return [], []

        # --- Prediction ---
        try: results = self.model.predict(source=img, conf=conf, verbose=False)
        except Exception as e: logger.error(f"Error during YOLO prediction for {img_path}: {e}"); return [], []

        # --- Detection Extraction ---
        all_detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy(); classes = r.boxes.cls.cpu().numpy(); confidences = r.boxes.conf.cpu().numpy()
            for box, cls, conf_score in zip(boxes, classes, confidences):
                detected_cls_id = int(cls)
                if detected_cls_id in self.target_class_ids:
                    all_detections.append({'box': box, 'cls': detected_cls_id, 'conf': float(conf_score), 'size': self._calculate_area(box)})
        if not all_detections: logger.info(f"No target objects detected in {img_path.name} with confidence >= {conf}."); return [], []

        # --- Sorting & Assign Per-Category Counter (pcnr) ---
        if len(all_detections) > 1 or self.process_single: all_detections = self._sort_detections(all_detections)
        category_counters = defaultdict(int)
        for det in all_detections:
            category_name = self.class_id_to_name.get(det['cls'], f"unknown_{det['cls']}")
            category_counters[category_name] += 1; det['pcnr'] = category_counters[category_name]

        # --- Determine which detections to process ---
        detections_to_process = all_detections[0:1] if self.process_single else all_detections
        logger.debug(f"Selected {len(detections_to_process)} detections to crop for {img_path.name}.")

        # --- Cropping and Saving Loop ---
        saved_crop_paths = []
        saved_metadata_paths = []
        for i, det in enumerate(detections_to_process):
            nr = i + 1; box = det['box']; detected_cls_id = det['cls']; pcnr = det['pcnr']

            # --- Calculate Coordinates with Margin ---
            x1_orig, y1_orig, x2_orig, y2_orig = map(int, box)
            x1, y1 = x1_orig - self.margin, y1_orig - self.margin; x2, y2 = x2_orig + self.margin, y2_orig + self.margin
            x1_crop, y1_crop = max(0, x1), max(0, y1); x2_crop, y2_crop = min(img_width, x2), min(img_height, y2)
            if x1_crop >= x2_crop or y1_crop >= y2_crop:
                logger.warning(f"Invalid crop dimensions after margin/clamping for detection {nr} (pcnr {pcnr}) in {img_path.name}, skipping."); continue

            try:
                # --- Perform Cropping ---
                # Crop only if not in dry run (to save computation) or if needed for metadata size
                crop = None
                if not dry_run or save_metadata: # Need dimensions even for dry run if saving metadata
                    crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
                    if crop.size == 0: logger.warning(f"Empty crop for detection {nr} (pcnr {pcnr}) in {img_path.name}, skipping."); continue
                    crop_height, crop_width = crop.shape[:2]
                else: # Dry run without metadata, just estimate dimensions roughly if needed
                    crop_height = max(0, y2_crop - y1_crop)
                    crop_width = max(0, x2_crop - x1_crop)


                # --- Prepare Data for Templating ---
                category_name = self.class_id_to_name.get(detected_cls_id, f"unknown_{detected_cls_id}")
                template_data = {
                    'p': img_path, 'stat': stat_info, 'exif': exif_info,
                    'box': det['box'], 'cls': detected_cls_id, 'category': category_name,
                    'conf': det['conf'], 'size': det['size'],
                    'x1': x1_orig, 'y1': y1_orig, 'x2': x2_orig, 'y2': y2_orig,
                    'nr': nr, 'pcnr': pcnr, 'width': crop_width, 'height': crop_height, 'margin': self.margin,
                    'x1_crop': x1_crop, 'y1_crop': y1_crop, 'x2_crop': x2_crop, 'y2_crop': y2_crop,
                }
                if stat_info: template_data['st_size'] = stat_info.st_size; template_data['st_mtime'] = stat_info.st_mtime
                if exif_info: template_data['DateTimeOriginal'] = exif_info.get('DateTimeOriginal', ''); template_data['Make'] = exif_info.get('Make', ''); template_data['Model'] = exif_info.get('Model', '')

                # --- Generate Output Path for Crop ---
                output_crop_path = generate_output_path(output_template, template_data, img_path)

                # --- Handle Crop Overwrite/Collision ---
                crop_write_allowed = False
                if force_overwrite or not output_crop_path.exists():
                    crop_write_allowed = True
                else:
                    logger.warning(f"Output crop file exists: {output_crop_path}. Skipping (use --force to overwrite).")

                # --- Save the crop OR Simulate ---
                if crop_write_allowed:
                    if dry_run:
                        logger.info(f"[DRY RUN] Would save crop: {output_crop_path} (Class: {category_name} #{pcnr}, Conf: {det['conf']:.2f}, Size: {det['size']:.0f}px)")
                        saved_crop_paths.append(output_crop_path) # Add to list even in dry run
                    else:
                        if crop is None: # Should not happen if not dry run, but safety check
                             logger.error(f"Internal error: Crop data is missing for {output_crop_path}. Skipping save.")
                             continue
                        success = cv2.imwrite(str(output_crop_path), crop)
                        if success:
                            logger.info(f"Saved crop: {output_crop_path} (Class: {category_name} #{pcnr}, Conf: {det['conf']:.2f}, Size: {det['size']:.0f}px)")
                            saved_crop_paths.append(output_crop_path)
                        else:
                            logger.error(f"Failed to write crop file: {output_crop_path}")

                # --- Handle Metadata Saving ---
                if save_metadata:
                    # Determine metadata path (same name as crop, but .json)
                    output_metadata_path = output_crop_path.with_suffix(".json")

                    # Check overwrite for metadata file separately
                    metadata_write_allowed = False
                    if force_overwrite or not output_metadata_path.exists():
                        metadata_write_allowed = True
                    else:
                         logger.warning(f"Output metadata file exists: {output_metadata_path}. Skipping (use --force to overwrite).")

                    if metadata_write_allowed:
                        if dry_run:
                            logger.info(f"[DRY RUN] Would save metadata to: {output_metadata_path}")
                            saved_metadata_paths.append(output_metadata_path) # Add to list even in dry run
                        else:
                            try:
                                # Ensure parent directory exists (generate_output_path might only ensure crop's parent)
                                output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
                                # Prepare data for JSON - use template_data, maybe simplify/filter?
                                # For now, save most of template_data. Exclude raw exif/stat objects if too complex.
                                metadata_to_save = {k: v for k, v in template_data.items() if k not in ['exif', 'stat']}
                                # Add back simplified versions if needed
                                # metadata_to_save['stat_simplified'] = { ... }
                                # metadata_to_save['exif_simplified'] = { ... }

                                with open(output_metadata_path, 'w', encoding='utf-8') as f:
                                    json.dump(metadata_to_save, f, cls=NpEncoder, indent=4) # Use custom encoder
                                logger.info(f"Saved metadata: {output_metadata_path}")
                                saved_metadata_paths.append(output_metadata_path)
                            except OSError as e:
                                 logger.error(f"Failed to create directory for metadata file {output_metadata_path}: {e}")
                            except Exception as e:
                                logger.error(f"Failed to write metadata file {output_metadata_path}: {e}", exc_info=True)

            except Exception as e:
                logger.exception(f"Error during processing loop for {img_path.name} (detection {nr}, pcnr {pcnr}): {e}")

        return saved_crop_paths, saved_metadata_paths # Return both lists

