#!/usr/bin/env python3
# run_birdcrop.py
"""
Command-line script to detect and crop objects from images using the birdcrop library.
"""

import argparse
import logging
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Set, Dict, Any # Added Dict, Any

# Import from the library
from birdcrop import BirdCropper, find_image_files
# from birdcrop.exceptions import BirdCropError

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# logging.getLogger("ultralytics").setLevel(logging.WARNING)
logger = logging.getLogger("run_birdcrop") # Use a specific name


# Removed prepare_tasks function - no longer needed

def main():
    """Parses arguments and runs the bird cropping process."""
    default_workers = min(8, os.cpu_count() + 4 if os.cpu_count() else 4)

    # --- Default Output Template ---
    # A sensible default that puts crops in a 'cropped' subdir relative to the input
    default_output_template = "{p.parent}/cropped/{p.stem}_crop_{nr}.jpg"
    default_single_output_template = "{p.parent}/cropped/{p.stem}.jpg"

    parser = argparse.ArgumentParser(
        description="Detect and crop objects (e.g., birds) from images using the birdcrop library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Input Arguments ---
    parser.add_argument(
        "inputs", nargs='*',
        help="Paths to input image files or directories."
    )
    parser.add_argument(
        "--input", "-i", type=str, action='append', default=[],
        help="Specify an input file or directory (can be used multiple times)."
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Recursively search for images in input directories."
    )
    # --- Output Arguments ---
    parser.add_argument(
        "--output-template", "-o", type=str, default=default_output_template,
        help="Output path template (Python str.format_map syntax). "
             "Available keys include: p (input Path object), stat (os.stat object), "
             "exif (dict), box, cls, conf, size, x1, y1, x2, y2 (detection details), "
             "nr (crop number), width, height (crop dimensions), margin, "
             "st_size, st_mtime, DateTimeOriginal, Make, Model, etc. "
             "Relative paths are anchored to the input image's directory. "
             "Default suffix is .jpg if not specified in template."
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force overwrite existing output files. If not set, existing files will be skipped."
    )
    # --- Model & Detection Arguments ---
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Path to the YOLOv8 model file (.pt)."
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold for detection (0.0 to 1.0)."
    )
    parser.add_argument(
        "--class-id", type=int, default=14,
        help="Class ID for the target object (default: 14 for 'bird' in COCO)."
    )
    parser.add_argument(
        "--margin", type=int, default=0,
        help="Pixel margin to add around the detected bounding box before cropping."
    )
    # --- Processing Arguments ---
    parser.add_argument(
        "--multiple", dest='single', action='store_false',
        help="Process and save ALL detected objects per image. Default is to save only the best one."
    )
    parser.add_argument(
        "--sortby", type=str, default="size", choices=["confidence", "size"],
        help="Criterion to sort detections ('confidence' or 'size'). Determines the 'best' object in single mode."
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=default_workers,
        help="Number of parallel worker threads for processing images."
    )
    parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help="Increase logging verbosity (e.g., -v for DEBUG, default INFO)."
    )

    args = parser.parse_args()
    if args.single and args.output_template == default_output_template:
        # If single mode is on, use a different default template to avoid overwriting
        args.output_template = default_single_output_template

    # --- Adjust Log Level ---
    log_level = logging.INFO
    if args.verbose == 1:
        log_level = logging.DEBUG
    elif args.verbose > 1:
        log_level = logging.DEBUG # Keep it at DEBUG for -vv or more for now
        # You could potentially set library loggers lower here if needed
        # logging.getLogger("birdcrop").setLevel(logging.DEBUG)

    # Set the root logger level
    logging.getLogger().setLevel(log_level)
    if log_level == logging.DEBUG:
        logger.debug("Debug logging enabled.")


    # --- Validate and Find Inputs ---
    all_input_paths_str = args.inputs + args.input
    if not all_input_paths_str:
        parser.error("No input files or directories specified.")

    logger.info("Searching for image files...")
    image_files_to_process = find_image_files(all_input_paths_str, args.recursive)

    if not image_files_to_process:
        logger.info("Exiting: No images found to process.")
        exit(0)

    # --- Log Configuration ---
    logger.info(f"Processing {len(image_files_to_process)} image(s).")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Target class ID: {args.class_id}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"Margin: {args.margin}px")
    logger.info(f"Process single best detection per image: {args.single}")
    if args.single or len(image_files_to_process) > 1:
        logger.info(f"Sorting criterion: {args.sortby}")
    logger.info(f"Output template: {args.output_template}")
    logger.info(f"Force overwrite: {args.force}")
    logger.info(f"Number of workers: {args.workers}")

    # --- Initialize Cropper (Load Model Once) ---
    try:
        logger.info("Loading detection model...")
        cropper = BirdCropper(
            model_path=args.model,
            process_single=args.single,
            sort_by=args.sortby,
            class_id=args.class_id,
            margin=args.margin # Pass margin
        )
    except ValueError as e: # Catch specific init errors
        logger.error(f"Configuration error: {e}")
        exit(1)
    except Exception as e: # Catch model loading errors
        logger.error(f"Failed to initialize BirdCropper: {e}", exc_info=log_level <= logging.DEBUG)
        logger.error("Exiting due to model loading failure.")
        exit(1)

    # --- Process Images Concurrently ---
    logger.info("Starting image processing...")
    start_time = time.time()
    total_crops_saved = 0
    processed_files_count = 0
    # Store futures -> input path mapping
    futures_map: Dict[Any, Path] = {}

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='Worker') as executor:
        # Submit tasks: Pass necessary args to detect_and_crop
        for img_path in image_files_to_process:
            future = executor.submit(
                cropper.detect_and_crop,
                img_path,
                args.conf,
                args.output_template, # Pass template
                args.force          # Pass force flag
            )
            futures_map[future] = img_path

        # Process results as they complete
        for future in as_completed(futures_map):
            img_path = futures_map[future]
            processed_files_count += 1
            try:
                saved_crop_paths = future.result()
                if saved_crop_paths:
                    # Logging is now done inside detect_and_crop per file
                    total_crops_saved += len(saved_crop_paths)

            except Exception as exc:
                # Catch errors raised from detect_and_crop or unexpected issues
                logger.error(f"An error occurred processing {img_path.name}: {exc}", exc_info=log_level <= logging.DEBUG)

            # Progress indicator
            if processed_files_count % 20 == 0 or processed_files_count == len(image_files_to_process):
                 # Log progress less frequently for large batches
                 logger.info(f"Progress: {processed_files_count}/{len(image_files_to_process)} images processed.")


    end_time = time.time()
    duration = end_time - start_time

    # --- Final Summary ---
    logger.info("-" * 30)
    logger.info(f"Processing Summary:")
    logger.info(f"  Processed {processed_files_count}/{len(image_files_to_process)} images.")
    logger.info(f"  Saved {total_crops_saved} crop(s).")
    logger.info(f"  Output paths generated using template: {args.output_template}")
    logger.info(f"  Total time: {duration:.2f} seconds")
    logger.info("-" * 30)

if __name__ == "__main__":
    main()
