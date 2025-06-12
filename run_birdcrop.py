# run_birdcrop.py
#!/usr/bin/env python3
"""
Command-line script to detect and crop objects from images using the birdcrop library.
"""

SCRIPT_VERSION = "0.2.4"
SCRIPT_DATE = "2025-06-12"

# Model size mapping for YOLOv8
YOLOV8_MODEL_SIZES = {
    "nano":   ("yolov8n.pt", "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"),
    "small":  ("yolov8s.pt", "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt"),
    "medium": ("yolov8m.pt", "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt"),
    "large":  ("yolov8l.pt", "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt"),
    "xlarge": ("yolov8x.pt", "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt"),
}

DEFAULT_MODEL_SIZE = "small"
DEFAULT_MODEL_FILENAME, DEFAULT_MODEL_URL = YOLOV8_MODEL_SIZES[DEFAULT_MODEL_SIZE]

# -------------------------------------------------------------------------- #

import argparse
import logging
import time
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Set, Dict, Any

# Import from the library
from ultralytics import YOLO
from birdcrop import BirdCropper, find_image_files
# from birdcrop.exceptions import BirdCropError

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# logging.getLogger("ultralytics").setLevel(logging.WARNING)
logger = logging.getLogger("run_birdcrop")


def parse_classes_arg(classes_str: str) -> List[str | int]:
    if not classes_str: return []
    items = []
    for item in classes_str.split(','):
        item = item.strip()
        if not item: continue
        if item.isdigit(): items.append(int(item))
        else: items.append(item)
    return items

def list_model_classes(model_path: str):
    logger.info(f"Loading model '{model_path}' to list classes...")
    try:
        model = YOLO(model_path)
        if not hasattr(model, 'names') or not isinstance(model.names, dict):
            logger.error(f"Model '{model_path}' loaded, but class names (model.names) are missing or not in the expected dictionary format.")
            sys.exit(1)
        print(f"\nClasses available in model '{model_path}':")
        print("-" * 40)
        for class_id, class_name in sorted(model.names.items()):
            print(f"  ID: {class_id:<5} Name: {class_name}")
        print("-" * 40)
        sys.exit(0)
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model '{model_path}' or access class names: {e}", exc_info=True)
        sys.exit(1)

def expand_input_lists(input_paths):
    """Expand any .csv/.lst/.txt files in input_paths into lists of image paths."""
    expanded = []
    for path in input_paths:
        p = Path(path)
        if p.suffix.lower() in {'.csv', '.lst', '.txt'} and p.is_file():
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        expanded.append(line)
        else:
            expanded.append(path)
    return expanded

import urllib.request

def download_model(model_path: str, url: str):
    resolved_path = os.path.abspath(model_path)
    logger.info(f"Model file '{resolved_path}' not found. Downloading from {url} ...")
    try:
        model_path_obj = Path(model_path)
        if model_path_obj.parent and not model_path_obj.parent.exists():
            model_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response, open(model_path, 'wb') as out_file:
            out_file.write(response.read())
        logger.info(f"Model downloaded successfully to '{model_path}'.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        sys.exit(1)

# -------------------------------------------------------------------------- #

def main():
    """Parses arguments and runs the bird cropping process."""
    default_workers = min(8, os.cpu_count() + 4 if os.cpu_count() else 4)

    # --- Default Output Templates ---
    default_output_template = "{p.parent}/{category}/{p.stem}_crop_{nr}.jpg"
    default_single_output_template = "{p.parent}/cropped/{p.stem}.jpg"

    parser = argparse.ArgumentParser(
        description="Detect and crop objects from images using the birdcrop library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Input Arguments ---
    parser.add_argument("inputs", nargs='*', help="Paths to input image files or directories. Required unless --list-classes is used.")
    parser.add_argument("--input", "-i", type=str, action='append', default=[], help="Specify an input file or directory (can be used multiple times).")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively search input directories for images.")
    # --- Output Arguments ---
    parser.add_argument("--output-template", "-o", type=str, help="Output path template (Python str.format_map syntax). Available keys include: p, stat, exif, box, cls (id), conf, size, x1, y1, x2, y2, nr (overall crop #), pcnr (per-category crop #), width, height, margin, category (name), etc. Relative paths are anchored to the input image's directory. Default for multiple crops: '{default_output_template}'. Default for single crop: '{default_single_output_template}'.")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing output files. If not set, existing files will be skipped.")
    # --- Model & Detection Arguments ---
    parser.add_argument("--model", type=str, default=None, help="Path to the YOLOv8 model file (.pt). If not specified, --model-size is used.")
    parser.add_argument("--model-size", type=str, choices=YOLOV8_MODEL_SIZES.keys(), default=DEFAULT_MODEL_SIZE,
                        help="YOLOv8 model size to use if --model is not specified. Choices: nano, small, medium, large, xlarge.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection (0.0 to 1.0).")
    # --- Class Specification ---
    parser.add_argument("--classes", type=str, default="bird", help='Comma-separated list of class names (e.g., "person,cat,dog") or class IDs (e.g., "0,15,16") to detect. Names are matched against the loaded model\'s class list.')
    parser.add_argument("--list-classes", action="store_true", help="List the classes available in the specified --model and exit.")
    parser.add_argument("--margin", type=int, default=5, help="Pixel margin to add around the detected bounding box before cropping.")
    # --- Processing Arguments ---
    parser.add_argument("--multiple", dest='single', action='store_false', help="Process and save ALL detected objects per image. Default is to save only the best one.")
    parser.add_argument("--sortby", type=str, default="size", choices=["confidence", "size"], help="Criterion to sort detections ('confidence' or 'size'). Determines the 'best' object in single mode.")
    parser.add_argument("--workers", "-w", type=int, default=default_workers, help="Number of parallel worker threads for processing images.")
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Increase logging verbosity (e.g., -v for DEBUG, default INFO).")
    # --- Miscellaneous Arguments ---
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate processing and show what files would be created without writing anything."
    )
    parser.add_argument(
        "--save-metadata", action="store_true",
        help="Save detection metadata (bounding box, confidence, etc.) as a JSON file alongside each crop."
    )
    parser.add_argument(
        "--do-not-preserve-exif", action="store_false", default=True, dest="preserve_exif",
        help="Do not preserve EXIF metadata from the original image in the cropped JPG/JPEG images."
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and date information for this script and the birdcrop library."
    )

    args = parser.parse_args()

    # --- Handle --version early ---
    if getattr(args, "version", False):
        import birdcrop
        print(f"run_birdcrop.py version: {SCRIPT_VERSION}\t(date: {SCRIPT_DATE})")
        print(f"birdcrop library version: {birdcrop.__version__}\t(date: {getattr(birdcrop, '__date__', 'unknown')})")
        sys.exit(0)

    # --- Handle --list-classes early ---
    if args.list_classes:
        list_model_classes(args.model)

    # --- Adjust Log Level ---
    log_level = logging.INFO
    if args.dry_run: # If dry run, always show INFO messages about what would happen
        log_level = logging.INFO
    elif args.verbose == 1:
        log_level = logging.DEBUG
    elif args.verbose > 1:
        log_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)
    if log_level == logging.DEBUG: logger.debug("Debug logging enabled.")
    if args.dry_run: logger.info("--- DRY RUN MODE ENABLED ---")


    # --- Set default output template ---
    if args.output_template is None:
        args.output_template = default_single_output_template if args.single else default_output_template

    # --- Parse --classes argument ---
    target_classes_input = parse_classes_arg(args.classes)
    if not target_classes_input:
        parser.error("No target classes specified or parsed from --classes argument.")

    # --- Validate and Find Inputs ---
    all_input_paths_str = expand_input_lists(args.inputs + args.input)
    if not all_input_paths_str:
        parser.error("No input files or directories specified (and --list-classes not used).")
    logger.info("Searching for image files...")
    image_files_to_process = find_image_files(all_input_paths_str, args.recursive)
    if not image_files_to_process:
        logger.info("Exiting: No images found to process."); exit(0)
    logger.info(f"Found {len(image_files_to_process)} image(s) to process.")

    # --- Log Configuration ---
    logger.info(f"Processing {len(image_files_to_process)} image(s).")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Target classes: {args.classes}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"Margin: {args.margin}px")
    logger.info(f"Process single best detection per image: {args.single}")
    if args.single or len(target_classes_input) > 1: logger.info(f"Sorting criterion: {args.sortby}")
    logger.info(f"Output template: {args.output_template}")
    logger.info(f"Force overwrite: {args.force}")
    logger.info(f"Save metadata: {args.save_metadata}") # Log new option
    logger.info(f"Preserve EXIF: {args.preserve_exif}")
    logger.info(f"Number of workers: {args.workers}")

    # --- Model selection and auto-download ---
    if args.model:
        model_path = args.model
        model_url = None
        logger.info(f"Using user-specified model: {model_path}")
    else:
        model_filename, model_url = YOLOV8_MODEL_SIZES[args.model_size]
        model_path = model_filename
        logger.info(f"No --model specified. Using --model-size '{args.model_size}': {model_filename}")

    if not os.path.isfile(model_path):
        if model_url:
            download_model(model_path, model_url)
        else:
            logger.error(f"Model file '{model_path}' not found and no auto-download URL is known for this file.")
            sys.exit(1)
            
    # --- Initialize Cropper ---
    try:
        logger.info("Loading detection model...")
        cropper = BirdCropper(
            model_path=model_path, target_classes=target_classes_input,
            process_single=args.single, sort_by=args.sortby, margin=args.margin
        )
        logger.info(f"Model '{args.model}' loaded. Targeting class IDs: {sorted(list(cropper.target_class_ids))}")
    except ValueError as e: logger.error(f"Configuration error: {e}"); exit(1)
    except FileNotFoundError as e: logger.error(f"Model file not found: {e}"); exit(1)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory! Try using a smaller model (e.g., --model-size nano or small), or run on CPU.")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize BirdCropper: {e}", exc_info=log_level <= logging.DEBUG)
        logger.error("Exiting due to model loading/initialization failure."); exit(1)

    # --- Process Images Concurrently ---
    logger.info("Starting image processing...")
    start_time = time.time()
    total_crops_saved = 0
    total_metadata_saved = 0 # Track metadata files
    processed_files_count = 0
    futures_map: Dict[Any, Path] = {}

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='Worker') as executor:
        for img_path in image_files_to_process:
            future = executor.submit(
                cropper.detect_and_crop,
                img_path,
                args.conf,
                args.output_template,
                args.force,
                args.dry_run, # Pass dry_run flag
                args.save_metadata, # Pass save_metadata flag
                preserve_exif=args.preserve_exif # Pass preserve_exif flag
            )
            futures_map[future] = img_path

        for future in as_completed(futures_map):
            img_path = futures_map[future]
            processed_files_count += 1
            try:
                # Result is now a tuple: (list_of_crop_paths, list_of_metadata_paths)
                saved_crop_paths, saved_metadata_paths = future.result()
                if saved_crop_paths: total_crops_saved += len(saved_crop_paths)
                if saved_metadata_paths: total_metadata_saved += len(saved_metadata_paths)
            except Exception as exc:
                logger.error(f"An error occurred processing {img_path.name}: {exc}", exc_info=log_level <= logging.DEBUG)

            if processed_files_count % 20 == 0 or processed_files_count == len(image_files_to_process):
                 logger.info(f"Progress: {processed_files_count}/{len(image_files_to_process)} images processed.")

    end_time = time.time()
    duration = end_time - start_time

    # --- Final Summary ---
    logger.info("-" * 30)
    logger.info(f"Processing Summary:")
    logger.info(f"  Mode: {'DRY RUN' if args.dry_run else 'Execution'}")
    logger.info(f"  Processed {processed_files_count}/{len(image_files_to_process)} images.")
    if args.dry_run:
        logger.info(f"  (Dry run: Would have potentially saved {total_crops_saved} crop(s) and {total_metadata_saved} metadata file(s))")
    else:
        logger.info(f"  Saved {total_crops_saved} crop(s).")
        if args.save_metadata:
            logger.info(f"  Saved {total_metadata_saved} metadata file(s).")
        if args.preserve_exif:
            logger.info(f"  (Attempted to preserve EXIF for saved crops)") # Actual count of preserved EXIF would depend on library
    logger.info(f"  Output paths generated using template: {args.output_template}")
    logger.info(f"  Total time: {duration:.2f} seconds")
    logger.info("-" * 30)

if __name__ == "__main__":
    main()
