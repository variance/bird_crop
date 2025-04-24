#!/usr/bin/env python3
# run_birdcrop.py
"""
Command-line script to detect and crop birds from images using the birdcrop library.
"""

import argparse
import logging
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Set

# Import from the library
from birdcrop import BirdCropper, find_image_files
# from birdcrop.exceptions import BirdCropError # Import base exception if using custom ones

# --- Logging Setup ---
# Configure logging for the application
# Library modules will use their own loggers, inheriting this configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(name)s - %(message)s', # Added logger name
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Adjust Ultralytics logging level if desired
# logging.getLogger("ultralytics").setLevel(logging.WARNING)
# Get a logger for the main script
logger = logging.getLogger(__name__)


def prepare_tasks(image_files: List[Path], output_arg: str | None) -> Tuple[List[Tuple[Path, Path]], Set[Path]]:
    """
    Determines the output directory for each image file and prepares task tuples.

    Args:
        image_files (List[Path]): List of input image paths.
        output_arg (str | None): The value of the --output argument.

    Returns:
        Tuple containing:
            - List[Tuple[Path, Path]]: Tasks as (input_image_path, output_directory_path).
            - Set[Path]: Unique output directories used.
    """
    tasks: List[Tuple[Path, Path]] = []
    output_dirs_used: Set[Path] = set()

    if output_arg:
        # User specified a single output directory for all images
        global_output_dir = Path(output_arg).resolve()
        try:
            # Create the directory here to fail early if it's invalid
            global_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using global output directory: {global_output_dir}")
            output_dirs_used.add(global_output_dir)
            for img_path in image_files:
                tasks.append((img_path, global_output_dir))
        except OSError as e:
            logger.error(f"Could not create specified output directory {global_output_dir}: {e}")
            # Optionally raise a custom exception or exit
            raise SystemExit(1) # Exit if the main output dir can't be made
    else:
        # Default: Use per-input 'cropped' subdirectory
        logger.info("Output directory not specified. Using 'cropped' subdirectory within each image's parent folder.")
        for img_path in image_files:
            # We expect BirdCropper.detect_and_crop to create this directory
            specific_output_dir = img_path.parent / "cropped"
            tasks.append((img_path, specific_output_dir))
            output_dirs_used.add(specific_output_dir) # Track unique dirs

    return tasks, output_dirs_used


def main():
    """Parses arguments and runs the bird cropping process."""
    # Determine a sensible default for workers
    default_workers = min(8, os.cpu_count() + 4 if os.cpu_count() else 4)

    parser = argparse.ArgumentParser(
        description="Detect and crop birds from images or directories using the birdcrop library.",
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
        "--output", "-o", type=str, default=None,
        help="Path to a SINGLE output directory for ALL cropped images. If not specified, crops are saved to a 'cropped' subdirectory within each input image's parent folder."
    )
    # --- Model & Detection Arguments ---
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Path to the YOLOv8 model file (.pt)."
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold for bird detection (0.0 to 1.0)."
    )
    parser.add_argument(
        "--class-id", type=int, default=14,
        help="Class ID for the target object (default: 14 for 'bird' in COCO)."
    )
    # --- Processing Arguments ---
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
    parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help="Increase logging verbosity (e.g., -v for DEBUG, default INFO)."
    )


    args = parser.parse_args()

    # --- Adjust Log Level ---
    if args.verbose == 1:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")
    elif args.verbose > 1:
        # Potentially enable even more verbose logging if needed in the future
         logging.getLogger().setLevel(logging.DEBUG)
         # You might want to lower library log levels too, e.g.:
         # logging.getLogger("birdcrop").setLevel(logging.DEBUG)
         logger.debug("Verbose debug logging enabled.")


    # --- Validate and Find Inputs ---
    all_input_paths_str = args.inputs + args.input
    if not all_input_paths_str:
        parser.error("No input files or directories specified.")

    logger.info("Searching for image files...")
    image_files_to_process = find_image_files(all_input_paths_str, args.recursive)

    if not image_files_to_process:
        # find_image_files already logs a warning
        logger.info("Exiting: No images found to process.")
        exit(0)

    # --- Prepare Tasks ---
    try:
        tasks, output_dirs_used = prepare_tasks(image_files_to_process, args.output)
    except SystemExit:
        exit(1) # Exit if prepare_tasks failed (e.g., couldn't create global output dir)


    # --- Log Configuration ---
    logger.info(f"Processing {len(image_files_to_process)} image(s).")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Target class ID: {args.class_id}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"Process single best detection per image: {args.single}")
    if args.single or len(image_files_to_process) > 1: # Log sortby if relevant
        logger.info(f"Sorting criterion: {args.sortby}")
    logger.info(f"Number of workers: {args.workers}")

    # --- Initialize Cropper (Load Model Once) ---
    try:
        logger.info("Loading detection model...")
        cropper = BirdCropper(
            model_path=args.model,
            process_single=args.single,
            sort_by=args.sortby,
            class_id=args.class_id
        )
    except Exception as e: # Catch potential model loading errors (or specific ModelLoadError)
        logger.error(f"Failed to initialize BirdCropper: {e}", exc_info=True)
        logger.error("Exiting due to model loading failure.")
        exit(1)

    # --- Process Images Concurrently ---
    logger.info("Starting image processing...")
    start_time = time.time()
    total_crops_saved = 0
    processed_files_count = 0
    futures_map = {} # Keep track of futures to image paths for error reporting

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='Worker') as executor:
        # Submit tasks
        for img_path, out_dir in tasks:
            future = executor.submit(cropper.detect_and_crop, img_path, out_dir, args.conf)
            futures_map[future] = img_path

        # Process results as they complete
        for future in as_completed(futures_map):
            img_path = futures_map[future]
            processed_files_count += 1
            try:
                saved_crop_paths = future.result() # Get the list of saved paths
                if saved_crop_paths:
                    logger.debug(f"Successfully processed: {img_path.name} -> {len(saved_crop_paths)} crop(s)")
                    total_crops_saved += len(saved_crop_paths)
                # else: # No crops saved (logged within detect_and_crop)
                #    logger.debug(f"Finished processing {img_path.name} (no crops saved).")

            except Exception as exc:
                # Catch errors raised from detect_and_crop or unexpected issues
                logger.error(f"An error occurred processing {img_path.name}: {exc}", exc_info=True)
                # Continue processing other images

            # Optional: Progress indicator
            if processed_files_count % 10 == 0 or processed_files_count == len(image_files_to_process):
                 logger.info(f"Progress: {processed_files_count}/{len(image_files_to_process)} images processed.")


    end_time = time.time()
    duration = end_time - start_time

    # --- Final Summary ---
    logger.info("-" * 30)
    logger.info(f"Processing Summary:")
    logger.info(f"  Processed {processed_files_count}/{len(image_files_to_process)} images.")
    logger.info(f"  Saved {total_crops_saved} crop(s).")
    if args.output:
        logger.info(f"  Output directory: {Path(args.output).resolve()}")
    else:
        logger.info(f"  Output crops saved to 'cropped' subdirectories within respective input folders.")
        # Optional: List unique output dirs if not too many
        # if len(output_dirs_used) < 10:
        #    logger.info("  Output directories used:")
        #    for odir in sorted(list(output_dirs_used)):
        #        logger.info(f"    - {odir}")
    logger.info(f"  Total time: {duration:.2f} seconds")
    logger.info("-" * 30)

if __name__ == "__main__":
    main()

