"""Command-line interface for BirdCrop."""
#!/usr/bin/env python3
"""
Command-line script to detect and crop objects from images using the birdcrop library.
"""

SCRIPT_VERSION = "0.3.8"
SCRIPT_DATE = "2026-10-08"

# -------------------------------------------------------------------------- #

import argparse
import importlib.metadata
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Import from the library
from ultralytics import YOLO

from birdcrop import BirdCropper, find_image_files
from birdcrop.utils import (_YOLO_RELEASE_URL_PREFIX, DEFAULT_MODEL_DIR,
                            DEFAULT_MODEL_SIZE, YOLO_MODEL_SIZES,
                            find_best_local_model)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# logging.getLogger("ultralytics").setLevel(logging.WARNING)
logger = logging.getLogger("birdcrop")


def _parse_version_parts(value: str) -> Tuple[int, ...]:
    """Extract numeric version parts from strings like '8.3.10' or '8.3.10rc1'."""
    parts = re.findall(r"\d+", value)
    if not parts:
        return (0,)
    return tuple(int(p) for p in parts[:4])


def _fetch_json(url: str, timeout: float = 2.0) -> Dict[str, Any] | None:
    """Fetch JSON from URL with a short timeout; return None on network/API errors."""
    req = Request(url, headers={"User-Agent": "birdcrop-update-check/1.0"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def check_ultralytics_update(timeout: float = 2.0):
    """Check PyPI for newer ultralytics package versions."""
    try:
        local_version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        logger.debug("Update check: ultralytics package not found in environment.")
        return
    except Exception as exc:
        logger.debug(f"Update check: could not read local ultralytics version: {exc}")
        return

    payload = _fetch_json("https://pypi.org/pypi/ultralytics/json", timeout=timeout)
    if not payload:
        logger.debug("Update check: could not query PyPI for ultralytics.")
        return

    latest_version = str(payload.get("info", {}).get("version", "")).strip()
    if not latest_version:
        logger.debug("Update check: PyPI response did not contain ultralytics version.")
        return

    if _parse_version_parts(latest_version) > _parse_version_parts(local_version):
        logger.warning(
            "ultralytics update available: local=%s latest=%s (PyPI)",
            local_version,
            latest_version,
        )
    else:
        logger.info("ultralytics is up to date (local=%s, latest=%s).", local_version, latest_version)


def check_model_release_update(current_assets_tag: str, model_filename: str | None = None, timeout: float = 2.0):
    """Check GitHub releases for newer ultralytics/assets model tags."""
    payload = _fetch_json("https://api.github.com/repos/ultralytics/assets/releases/latest", timeout=timeout)
    if not payload:
        logger.debug("Update check: could not query GitHub latest ultralytics/assets release.")
        return

    latest_tag = str(payload.get("tag_name", "")).strip()
    if not latest_tag:
        logger.debug("Update check: GitHub response did not include tag_name.")
        return

    if _parse_version_parts(latest_tag) <= _parse_version_parts(current_assets_tag):
        logger.info("Model assets tag is up to date (configured=%s, latest=%s).", current_assets_tag, latest_tag)
        return

    assets = payload.get("assets") or []
    if model_filename:
        has_matching_asset = any(a.get("name") == model_filename for a in assets)
        if has_matching_asset:
            logger.warning(
                "Newer model release available for %s: configured tag=%s, latest tag=%s.",
                model_filename,
                current_assets_tag,
                latest_tag,
            )
        else:
            logger.warning(
                "Newer ultralytics/assets release detected (configured tag=%s, latest tag=%s), "
                "but '%s' is not listed in latest assets.",
                current_assets_tag,
                latest_tag,
                model_filename,
            )
    else:
        logger.warning(
            "Newer ultralytics/assets release detected: configured tag=%s, latest tag=%s.",
            current_assets_tag,
            latest_tag,
        )


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
    cpu_count = os.cpu_count() or 0
    default_workers = min(8, cpu_count + 4)
    cli_args_str = " ".join(sys.argv[1:])

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
    parser.add_argument("--output-template", "-o", type=str, help="Output path template (Python str.format_map syntax). Available keys include:"
                         " p, stat, exif, box, cls (id), conf, size, x1, y1, x2, y2, nr (overall crop #), pcnr (per-category crop #), width, height, margin, category (name), etc."
                         " Relative paths are anchored to the input image's directory."
                         f" Default for multiple crops or multiple classes: '{default_output_template}' else '{default_single_output_template}'.")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing output files. If not set, existing files will be skipped.")
    # --- Model & Detection Arguments ---
    parser.add_argument("--model", type=str, default=None, help="Path to the YOLO model file (yolo???.pt)."
                        f" If not specified, the --model-size and the default model directory '{DEFAULT_MODEL_DIR}' will be used.")
    parser.add_argument("--model-size", type=str, choices=YOLO_MODEL_SIZES.keys(), default=None,
                        help=f"YOLO model size to use if --model is not specified. Defaults to '{DEFAULT_MODEL_SIZE}' if no local model is found.")
    parser.add_argument("--confidence", '-C', type=float, default=0.5, help="Confidence threshold for detection (0.0 to 1.0). Default: 0.5.")
    # --- Class Specification ---
    parser.add_argument("--classes", '-c', type=str, default="bird", help='Comma-separated list of class names (e.g., "person,car,cat,dog") or class IDs (e.g., "0,2,15,16") to detect.'
                        ' Names are matched against the loaded model\'s class list.')
    parser.add_argument("--list-classes", '-l', action="store_true", help="List the classes available in the specified --model and exit.")
    parser.add_argument("--margin", type=int, default=5, help="Pixel margin to add around the detected bounding box before cropping.")
    # --- Processing Arguments ---
    parser.add_argument("--multiple", "-m", dest='single', action='store_false', help="Process and save ALL detected objects per image. Default is to save only the best one.")
    parser.add_argument("--sortby", type=str, default="size", choices=["confidence", "size"], help="Criterion to sort detections ('confidence' or 'size'). Determines the 'best' object in single mode.")
    parser.add_argument("--workers", "-w", type=int, default=default_workers, help="Number of parallel worker threads for processing images.")
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Increase logging verbosity (e.g., -v for DEBUG, default INFO).")
    # --- Miscellaneous Arguments ---
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate processing and show what files would be created without writing anything."
    )
    parser.add_argument(
        "--save-metadata", '-M', action="store_true",
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
    parser.add_argument(
        "--check-updates", dest="check_updates", action="store_true", default=True,
        help="Check PyPI/GitHub for newer ultralytics package and model release tags."
    )
    parser.add_argument(
        "--no-update-check", dest="check_updates", action="store_false",
        help="Disable online update checks for ultralytics package/model releases."
    )
    parser.add_argument(
        "--update-check-timeout", type=float, default=2.0,
        help="Timeout in seconds for each online update check request."
    )
    if sys.platform == "win32":
        parser.add_argument(
            "--shortcut", action="store_true",
            help="Indicates a desktop shortcut was used to launch the script."
        )

    args = parser.parse_args()

    # --- Handle --version early ---
    if getattr(args, "version", False):
        import birdcrop
        print(f"birdcrop command version: {SCRIPT_VERSION}\t(date: {SCRIPT_DATE})")
        print(f"birdcrop library version: {birdcrop.__version__}\t(date: {getattr(birdcrop, '__date__', 'unknown')})")
        sys.exit(0)

    # --- Handle --list-classes early ---
    if args.list_classes:
        if not args.model:
            parser.error("--list-classes requires --model to be specified (path to a 'yolo<VERSION>.pt' file).")
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


    # --- Parse --classes argument ---
    if args.classes and args.classes.strip().lower() == "all":
        logger.info("Target classes set to 'ALL'. All classes in the model will be processed.")
        target_classes_input: List[str | int] = [ "ALL" ]
        args.all_classes = True
    else:
        target_classes_input: List[str | int] = parse_classes_arg(args.classes)
        args.all_classes = False
    if not target_classes_input:
        parser.error("No target classes specified or parsed from --classes argument.")

    # --- Set default output template ---
    if args.output_template is None:
        args.output_template = default_single_output_template if args.single and len(target_classes_input) == 1 and not args.all_classes else default_output_template

    # --- Validate and Find Inputs ---
    all_input_paths_str = expand_input_lists(args.inputs + args.input)
    if not all_input_paths_str:
        if sys.platform == "win32" and getattr(args, "shortcut", False):
            print(f"""BirdCrop Drag & Drop
            
            Usage: drag one or more image files or folders onto this file!
            Example: select a folder in Explorer and drag it onto the BirdCrop shortcut.
            
            To process a folder from a command prompt, use:
            birdcrop "C:\\path\\to\\images"
            Use birdcrop --help for more options.
            Shortcut command line arguments: {cli_args_str}
            """)
            input("Press Enter to continue...")
            sys.exit(0)
        else:
            parser.error("No input files or directories specified (and --list-classes not used).")
    logger.info("Searching for image files...")
    image_files_to_process = find_image_files(all_input_paths_str, args.recursive)
    if not image_files_to_process:
        logger.info("Exiting: No images found to process."); exit(0)
    logger.info(f"Found {len(image_files_to_process)} image(s) to process.")

    # --- Log Configuration ---
    logger.info(f"Processing {len(image_files_to_process)} image(s).")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Target classes: {'ALL' if args.all_classes else args.classes}")
    logger.info(f"Confidence threshold: {args.confidence}")
    logger.info(f"Margin: {args.margin}px")
    logger.info(f"Process single best detection per image: {args.single}")
    if not args.single or len(target_classes_input) > 1 or args.all_classes:
        logger.info(f"Sorting criterion: {args.sortby}")
    logger.info(f"Output template: {args.output_template}")
    logger.info(f"Force overwrite: {args.force}")
    logger.info(f"Save metadata: {args.save_metadata}") # Log new option
    logger.info(f"Preserve EXIF: {args.preserve_exif}")
    logger.info(f"Number of workers: {args.workers}")

    # --- Model selection and auto-download ---
    model_url = None

    if args.model:
        # Fall 1: Benutzer hat explizit einen Pfad mit --model angegeben
        model_path = args.model
        logger.info(f"Using user-specified model: {model_path}")
        configured_assets_tag = _YOLO_RELEASE_URL_PREFIX.rstrip('/').split('/')[-1]
        selected_model_filename = Path(model_path).name

    elif args.model_size:
        # Fall 2: Benutzer hat explizit eine Modellgröße via --model-size gewählt
        model_filename, model_url = YOLO_MODEL_SIZES[args.model_size]
        model_path = str(DEFAULT_MODEL_DIR / model_filename)
        logger.info(f"Using explicitly requested --model-size '{args.model_size}': {model_path}")
        configured_assets_tag = _YOLO_RELEASE_URL_PREFIX.rstrip('/').split('/')[-1]
        selected_model_filename = model_filename

    else:
        # Fall 3: Weder --model noch --model-size wurden angegeben -> Erst lokal suchen
        found_local = find_best_local_model(DEFAULT_MODEL_DIR)
        
        if found_local:
            model_filename, found_size = found_local
            model_path = str(DEFAULT_MODEL_DIR / model_filename)
            logger.info(f"No model specified. Found existing local model '{model_filename}' (size: {found_size}) in {DEFAULT_MODEL_DIR}")
            configured_assets_tag = _YOLO_RELEASE_URL_PREFIX.rstrip('/').split('/')[-1]
            selected_model_filename = model_filename
        else:
            # Fallback: Kein lokales Modell vorhanden -> Default-Modell (small) festlegen
            model_filename, model_url = YOLO_MODEL_SIZES[DEFAULT_MODEL_SIZE]
            model_path = str(DEFAULT_MODEL_DIR / model_filename)
            logger.info(f"No local model found in {DEFAULT_MODEL_DIR}. Defaulting to --model-size '{DEFAULT_MODEL_SIZE}': {model_path}")
            configured_assets_tag = _YOLO_RELEASE_URL_PREFIX.rstrip('/').split('/')[-1]
            selected_model_filename = model_filename

    if args.check_updates:
        if args.update_check_timeout <= 0:
            logger.warning("Skipping update check due to non-positive --update-check-timeout=%s", args.update_check_timeout)
        else:
            logger.info("Checking for updates (timeout %.1fs per endpoint)...", args.update_check_timeout)
            check_ultralytics_update(timeout=args.update_check_timeout)
            check_model_release_update(
                current_assets_tag=configured_assets_tag,
                model_filename=selected_model_filename,
                timeout=args.update_check_timeout,
            )
    else:
        logger.info("Update checks are disabled (--no-update-check).")

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
            process_single=args.single, sort_by=args.sortby, margin=args.margin, all_classes=args.all_classes
        )
        logger.info(f"Model '{model_path}' loaded. Targeting class IDs: {sorted(list(cropper.target_class_ids))}")
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
    output_dirs_created: Set[Path] = set()
    output_dirs_counter: Counter[str] = Counter() # basename of output directories created -> count of crops saved there

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='Worker') as executor:
        for img_path in image_files_to_process:
            future = executor.submit(
                cropper.detect_and_crop,
                img_path,
                args.confidence,
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
                if saved_crop_paths:
                    for path in saved_crop_paths:
                        output_dirs_created.add(path.parent)
                        output_dirs_counter[path.parent.name] += 1
                    total_crops_saved += len(saved_crop_paths)
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
    logger.info(f"  Model used: {model_path}")
    logger.info(f"  Confidence threshold: {args.confidence}")
    logger.info(f"  Target classes: {','.join(cropper.target_class_names)}")
    logger.info(f"  {'Single best detection' if args.single else 'All detections'} per image")
    logger.info(f"  Margin: {args.margin}px")
    logger.info(f"  Processed {processed_files_count}/{len(image_files_to_process)} images.")
    if args.dry_run:
        logger.info(f"  (Dry run: Would have potentially saved {total_crops_saved} crop(s) and {total_metadata_saved} metadata file(s))")
    else:
        logger.info(f"  Saved {total_crops_saved} crop(s).")
        logger.info(f"  Crops saved in each output directory: {', '.join(f'{name}: {count}' for name, count in sorted(output_dirs_counter.items()))}")
        if args.save_metadata:
            logger.info(f"  Saved {total_metadata_saved} metadata file(s).")
        if args.preserve_exif:
            logger.info(f"  (Attempted to preserve EXIF for saved crops)") # Actual count of preserved EXIF would depend on library
    logger.info(f"  Output paths generated using template: {args.output_template}")
    logger.info(f"  Output directories: {len(output_dirs_created) if len(output_dirs_created) != 1 else output_dirs_created.pop()}") # created unless already existing
    logger.info(f"  Total time: {duration:.2f} seconds")
    logger.info("-" * 30)

    if sys.platform == "win32" and getattr(args, "shortcut", False):
        input("Press Enter to continue...")  # Pause to allow user to see the summary when launched from a Windows shortcut

if __name__ == "__main__":
    main()
