# birdcrop/cropper.py
"""Contains the core BirdCropper class for detection and cropping."""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
from typing import List, Dict, Any

# Optional: Import custom exceptions if using them
# from .exceptions import ModelLoadError, ImageProcessingError, PredictionError, FileWriteError, DirectoryCreationError

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class BirdCropper:
    """
    Detects and crops birds from images using a YOLO model.
    Handles model loading, detection, sorting, and cropping logic.
    """
    def __init__(self, model_path: str = "yolov8n.pt", process_single: bool = True, sort_by: str = "size", class_id: int = 14):
        """
        Initializes the BirdCropper.

        Args:
            model_path (str): Path to the YOLOv8 model file.
            process_single (bool): If True, only process the single best bird per image.
            sort_by (str): Criterion ("confidence" or "size") to determine the best bird.
            class_id (int): The class ID to target for detection (default is 14 for 'bird' in COCO).

        Raises:
            ModelLoadError: If the YOLO model cannot be loaded.
        """
        self.model_path = model_path
        self.process_single = process_single
        self.sort_by = sort_by
        self.class_id = class_id

        if sort_by not in ["confidence", "size"]:
             raise ValueError("sort_by must be 'confidence' or 'size'")

        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.exception(f"Failed to load YOLO model from {self.model_path}: {e}")
            # raise ModelLoadError(f"Failed to load YOLO model from {self.model_path}: {e}") from e # Optional custom exception
            raise # Re-raise original exception for now

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

        if self.sort_by == "confidence":
            detections.sort(key=lambda x: x['conf'], reverse=True)
            logger.debug(f"Sorted {len(detections)} detections by confidence.")
        elif self.sort_by == "size":
            # Calculate size on the fly for sorting if not already present
            for det in detections:
                 if 'size' not in det: # Avoid recalculating if already done
                     det['size'] = self._calculate_area(det['box'])
            detections.sort(key=lambda x: x['size'], reverse=True)
            logger.debug(f"Sorted {len(detections)} detections by size.")
        return detections

    def detect_and_crop(self, img_path: Path, output_dir: Path, conf: float = 0.5) -> List[Path]:
        """
        Detects birds in a single image, crops them, and saves the crops.

        Args:
            img_path (Path): Path to the input image file.
            output_dir (Path): Directory to save cropped images for THIS specific image.
            conf (float): Confidence threshold for detection.

        Returns:
            list[Path]: A list of paths to the saved crop files.

        Raises:
            DirectoryCreationError: If the output directory cannot be created.
            ImageProcessingError: If the image cannot be read or processed.
            PredictionError: If the YOLO prediction fails.
            FileWriteError: If saving a crop fails.
        """
        # Ensure the specific output directory for this image exists before processing
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create output directory {output_dir} for {img_path.name}: {e}")
            # raise DirectoryCreationError(f"Could not create output directory {output_dir} for {img_path.name}: {e}") from e
            return [] # Return empty list if dir creation fails in this context

        if not img_path.is_file():
            logger.warning(f"Input path is not a file, skipping: {img_path}")
            return []

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
            # raise ImageProcessingError(f"Error reading or processing image {img_path}: {e}") from e
            return []

        # --- Prediction ---
        try:
            results = self.model.predict(source=img, conf=conf, verbose=False)
        except Exception as e:
            logger.error(f"Error during YOLO prediction for {img_path}: {e}")
            # raise PredictionError(f"Error during YOLO prediction for {img_path}: {e}") from e
            return []

        # --- Detection Extraction ---
        all_detections = []
        for r in results: # Usually one result for single image prediction
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            for box, cls, conf_score in zip(boxes, classes, confidences):
                 if int(cls) == self.class_id:
                    # Store necessary info, calculate size immediately for potential sorting
                    detection_info = {
                        'box': box,
                        'conf': float(conf_score),
                        'size': self._calculate_area(box) # Calculate size here
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
            box = det['box']
            conf_score = det['conf']
            box_area = det['size'] # Use pre-calculated size

            x1, y1, x2, y2 = map(int, box)
            # Clamp coordinates to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid crop dimensions [x1={x1}, y1={y1}, x2={x2}, y2={y2}] calculated for detection {i} in {img_path.name}, skipping.")
                continue

            try:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    logger.warning(f"Calculated crop resulted in empty image for detection {i} in {img_path.name}, skipping.")
                    continue

                # --- Filename Generation ---
                base_stem = img_path.stem
                if self.process_single:
                    # Use a consistent name for the single best crop
                    crop_filename_base = f"{base_stem}_best_crop"
                    crop_filename_ext = ".jpg" # Or derive from original? For simplicity, use jpg.
                else:
                    # Use index if processing multiple crops from the same image
                    crop_filename_base = f"{base_stem}_crop_{i}"
                    crop_filename_ext = ".jpg"

                crop_path = output_dir / f"{crop_filename_base}{crop_filename_ext}"

                # --- Filename Collision Handling ---
                counter = 0
                while crop_path.exists():
                    counter += 1
                    crop_filename = f"{crop_filename_base}_{counter}{crop_filename_ext}"
                    crop_path = output_dir / crop_filename
                    if counter > 100: # Safety break
                        logger.error(f"Too many filename collisions for {crop_filename_base} in {output_dir}, stopping crop attempts for this detection.")
                        # Optional: raise FileWriteError(...)
                        break # Break inner loop, go to next detection if any
                if counter > 100: continue # Skip saving if we hit the safety break

                # --- Save the crop ---
                success = cv2.imwrite(str(crop_path), crop)
                if success:
                    logger.info(f"Saved crop: {crop_path} (Confidence: {conf_score:.2f}, Size: {box_area:.0f}px)")
                    saved_crops.append(crop_path)
                else:
                    logger.error(f"Failed to write crop file: {crop_path}")
                    # Optional: raise FileWriteError(f"Failed to write crop file: {crop_path}")

            except Exception as e:
                # Catch potential errors during cropping (e.g., memory issues) or saving
                logger.exception(f"Error during cropping or saving for {img_path.name} (detection {i}): {e}")
                # Optional: raise FileWriteError(...) or ImageProcessingError(...) depending on the context

        return saved_crops

