import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List, Tuple, Set # Added Set for type hinting

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# logging.getLogger("ultralytics").setLevel(logging.WARNING)

# --- Constants ---
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

class BirdCropper:
    """
    Detects and crops birds from images using a YOLO model.
    """
    def __init__(self, model_path="yolov8n.pt", process_single=True, sort_by="size"):
        """
        Initializes the BirdCropper.

        Args:
            model_path (str): Path to the YOLOv8 model file.
            process_single (bool): If True, only process the single best bird per image.
            sort_by (str): Criterion ("confidence" or "size") to determine the best bird.
        """
        try:
            self.model = YOLO(model_path)
            logging.info(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            logging.exception(f"Failed to load YOLO model from {model_path}: {e}")
            raise
        self.class_id = 14
        self.process_single = process_single
        self.sort_by = sort_by

    def _calculate_area(self, box):
        """Calculates the area of a bounding box."""
        x1, y1, x2, y2 = box
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return width * height

    def detect_and_crop(self, img_path: Path, output_dir: Path, conf: float = 0.5) -> List[Path]:
        """
        Detects birds in a single image, crops them, and saves the crops.

        Args:
            img_path (Path): Path to the input image file.
            output_dir (Path): Directory to save cropped images for THIS specific image.
            conf (float): Confidence threshold for detection.

        Returns:
            list[Path]: A list of paths to the saved crop files.
        """
        # Ensure the specific output directory for this image exists before processing
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Could not create output directory {output_dir} for {img_path.name}: {e}")
            return [] # Cannot proceed without output dir

        if not img_path.is_file():
            logging.warning(f"Input path is not a file, skipping: {img_path}")
            return []

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logging.error(f"Could not read image: {img_path}")
                return []
            img_height, img_width = img.shape[:2]
            logging.debug(f"Processing image: {img_path} ({img_width}x{img_height})")

        except Exception as e:
            logging.error(f"Error reading or processing image {img_path}: {e}")
            return []

        try:
            results = self.model.predict(source=img, conf=conf, verbose=False)
        except Exception as e:
            logging.error(f"Error during YOLO prediction for {img_path}: {e}")
            return []

        saved_crops = []
        all_detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            for box, cls, conf_score in zip(boxes, classes, confidences):
                 if int(cls) == self.class_id:
                    all_detections.append({'box': box, 'conf': conf_score})

        if not all_detections:
            logging.info(f"No birds detected in {img_path.name} with confidence >= {conf}.")
            return []

        if len(all_detections) > 1 or self.process_single:
            if self.sort_by == "confidence":
                all_detections.sort(key=lambda x: x['conf'], reverse=True)
                logging.debug(f"Sorted {len(all_detections)} detections by confidence for {img_path.name}.")
            elif self.sort_by == "size":
                all_detections.sort(key=lambda x: self._calculate_area(x['box']), reverse=True)
                logging.debug(f"Sorted {len(all_detections)} detections by size for {img_path.name}.")

        detections_to_process = all_detections[0:1] if self.process_single else all_detections

        for i, det in enumerate(detections_to_process):
            box = det['box']
            conf_score = det['conf']
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            if x1 >= x2 or y1 >= y2:
                logging.warning(f"Invalid crop dimensions [x1={x1}, y1={y1}, x2={x2}, y2={y2}] calculated for a detection in {img_path.name}, skipping.")
                continue

            try:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    logging.warning(f"Calculated crop resulted in empty image for a detection in {img_path.name}, skipping.")
                    continue

                base_stem = img_path.stem
                if self.process_single:
                    crop_filename = f"{base_stem}.jpg"
                else:
                    crop_filename = f"{base_stem}_crop_{i}.jpg"

                crop_path = output_dir / crop_filename
                counter = 0
                while crop_path.exists():
                    counter += 1
                    if self.process_single:
                         crop_filename = f"{base_stem}_best_crop_{counter}.jpg"
                    else:
                         crop_filename = f"{base_stem}_crop_{i}_{counter}.jpg"
                    crop_path = output_dir / crop_filename
                    if counter > 100:
                        logging.error(f"Too many filename collisions for {base_stem} in {output_dir}, stopping.")
                        break
                if counter > 100: continue

                success = cv2.imwrite(str(crop_path), crop)
                if success:
                    box_area = self._calculate_area(box)
                    logging.info(f"Saved crop: {crop_path} (Confidence: {conf_score:.2f}, Size: {box_area:.0f}px)")
                    saved_crops.append(crop_path)
                else:
                    logging.error(f"Failed to write crop file: {crop_path}")

            except Exception as e:
                logging.exception(f"Error during cropping or saving for {img_path.name} (detection {i}): {e}")

        return saved_crops

# --- Helper Function for Input Processing ---
def find_image_files(input_paths: List[str], recursive: bool) -> List[Path]:
    """
    Finds all image files in the given paths (files or directories).

    Args:
        input_paths (list[str]): List of input file or directory paths.
        recursive (bool): Whether to search directories recursively.

    Returns:
        list[Path]: A list of Path objects for found image files.
    """
    image_files = []
    for path_str in input_paths:
        path = Path(path_str).resolve() # Resolve paths early for consistency
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_files.append(path)
            else:
                logging.warning(f"Skipping non-image file: {path}")
        elif path.is_dir():
            logging.info(f"Scanning directory: {path}{' (recursively)' if recursive else ''}")
            pattern = '**/*' if recursive else '*'
            for item in path.glob(pattern):
                if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_files.append(item)
        else:
            logging.warning(f"Input path is neither a file nor a directory, skipping: {path}")
    # Remove duplicates and sort
    return sorted(list(set(image_files)))

# --- Main Execution ---
if __name__ == "__main__":
    default_workers = min(8, os.cpu_count() + 4 if os.cpu_count() else 4)

    parser = argparse.ArgumentParser(
        description="Detect and crop birds from images or directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "inputs", nargs='*',
        help="Paths to input image files or directories."
    )
    parser.add_argument(
        "--input", type=str, action='append', default=[],
        help="Specify an input file or directory (can be used multiple times)."
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Recursively search for images in input directories."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to a SINGLE output directory for ALL cropped images. If not specified, crops are saved to a 'cropped' subdirectory within each input image's parent folder." # Clarified help text
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Path to the YOLOv8 model file (.pt)."
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold for bird detection."
    )
    parser.add_argument(
        "--multiple", dest='single', action='store_false',
        help="Process and save ALL detected birds per image. Default is to save only the best one."
    )
    parser.add_argument(
        "--sortby", type=str, default="size", choices=["confidence", "size"],
        help="Criterion to sort detections ('confidence' or 'size'). Determines the 'best' bird in single mode."
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=default_workers,
        help="Number of parallel worker threads for processing images."
    )

    args = parser.parse_args()

    all_input_paths_str = args.inputs + args.input
    if not all_input_paths_str:
        parser.error("No input files or directories specified.")

    image_files_to_process = find_image_files(all_input_paths_str, args.recursive)

    if not image_files_to_process:
        logging.warning("No processable image files found in the specified inputs.")
        exit(0)

    logging.info(f"Found {len(image_files_to_process)} image(s) to process.")
    logging.info(f"Using model: {args.model}")
    logging.info(f"Confidence threshold: {args.conf}")
    logging.info(f"Process single best bird per image: {args.single}")
    if args.single:
        logging.info(f"Sorting criterion for best bird: {args.sortby}")
    logging.info(f"Number of workers: {args.workers}")

    # --- Prepare Tasks (Image Path + Output Directory) ---
    tasks: List[Tuple[Path, Path]] = []
    output_dirs_used: Set[Path] = set() # Keep track of unique output dirs used

    if args.output:
        # User specified a single output directory for all images
        global_output_dir = Path(args.output).resolve()
        try:
            global_output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Using global output directory: {global_output_dir}")
            output_dirs_used.add(global_output_dir)
            for img_path in image_files_to_process:
                tasks.append((img_path, global_output_dir))
        except OSError as e:
            logging.error(f"Could not create specified output directory {global_output_dir}: {e}")
            exit(1)
    else:
        # Default: Use per-input 'cropped' subdirectory
        logging.info("Output directory not specified. Using 'cropped' subdirectory within each image's parent folder.")
        for img_path in image_files_to_process:
            specific_output_dir = img_path.parent / "cropped"
            tasks.append((img_path, specific_output_dir))
            output_dirs_used.add(specific_output_dir) # Will store unique dirs

    # --- Initialize Cropper (Load Model Once) ---
    try:
        cropper = BirdCropper(
            model_path=args.model,
            process_single=args.single,
            sort_by=args.sortby
        )
    except Exception:
        logging.error("Exiting due to model loading failure.")
        exit(1)

    # --- Process Images Concurrently ---
    start_time = time.time()
    total_crops_saved = 0
    processed_files = 0

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='Worker') as executor:
        # Submit tasks with specific output directories
        futures = {executor.submit(cropper.detect_and_crop, img_path, out_dir, args.conf): img_path
                   for img_path, out_dir in tasks} # Unpack the tuple here

        for future in as_completed(futures):
            img_path = futures[future]
            try:
                saved_crop_paths = future.result()
                if saved_crop_paths:
                    logging.debug(f"Successfully processed: {img_path.name} -> {len(saved_crop_paths)} crop(s)")
                    total_crops_saved += len(saved_crop_paths)
                processed_files += 1
            except Exception as exc:
                logging.error(f"An error occurred processing {img_path.name}: {exc}", exc_info=True)
                processed_files += 1

    end_time = time.time()
    duration = end_time - start_time

    logging.info("-" * 30)
    logging.info(f"Processing Summary:")
    logging.info(f"  Processed {processed_files}/{len(image_files_to_process)} images.")
    logging.info(f"  Saved {total_crops_saved} crop(s).")
    # Provide more specific info about output locations
    if args.output:
        logging.info(f"  Output directory: {global_output_dir}")
    else:
        logging.info(f"  Output crops saved to 'cropped' subdirectories within respective input folders.")
        # Optional: List the unique directories if needed (can be many)
        # logging.info("  Output directories used:")
        # for odir in sorted(list(output_dirs_used)):
        #     logging.info(f"    - {odir}")
    logging.info(f"  Total time: {duration:.2f} seconds")
    logging.info("-" * 30)
