#!/usr/bin/env python3
"""
Upgrade utility for BirdCrop: updates the ultralytics package and optionally downloads the latest YOLO models.
"""

SCRIPT_VERSION = "0.1.2"
SCRIPT_DATE = "2026-09-30"

import argparse
import logging
import subprocess
import sys
import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from typing import Dict, Any, List

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("upgrade_birdcrop")


def _fetch_json(url: str, timeout: float = 5.0) -> Dict[str, Any] | None:
    """Fetch JSON from URL with timeout; return None on errors."""
    req = Request(url, headers={"User-Agent": "birdcrop-upgrade/1.0"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
        logger.debug(f"Could not fetch {url}: {e}")
        return None


def upgrade_ultralytics_package():
    """Upgrade the ultralytics package via pip."""
    logger.info("Upgrading ultralytics package...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            logger.info("ultralytics package upgraded successfully.")
            if result.stdout:
                logger.debug(f"pip output: {result.stdout}")
            return True
        else:
            logger.error(f"Failed to upgrade ultralytics: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("pip upgrade timed out after 120 seconds.")
        return False
    except Exception as e:
        logger.error(f"Error during pip upgrade: {e}")
        return False



def download_latest_models(output_dir: str = ".", model_list: List[str] | None = None, model_major_version: str = "26") -> bool:
    """Download the latest YOLO models from the latest GitHub release.
    
    Args:
        output_dir (str): Directory to download models into.
        model_list (List[str] | None): List of model names to download. If None, downloads default models.
        model_major_version (str): The major version of the YOLO models to download (e.g., "26" for YOLO26 or "v8" for YOLOv8).
    Returns:
        bool: True if all models were downloaded successfully, False otherwise.
    """
    logger.info("Fetching latest ultralytics/assets release...")
    
    payload = _fetch_json("https://api.github.com/repos/ultralytics/assets/releases/latest", timeout=5.0)
    if not payload:
        logger.error("Could not fetch latest release info from GitHub.")
        return False
    
    tag_name = payload.get("tag_name", "").strip()
    if not tag_name:
        logger.error("Latest release tag not found in GitHub response.")
        return False
    
    logger.info(f"Latest release tag: {tag_name}")
    
    assets = payload.get("assets") or []
    if not assets:
        logger.error("No assets found in latest release.")
        return False
    
    # Filter to YOLO model files
    if model_list is None:
        # Default models
        v = model_major_version # for interpolation into the model names
        model_list = [f"yolo{v}n.pt", f"yolo{v}s.pt", f"yolo{v}m.pt", f"yolo{v}l.pt", f"yolo{v}x.pt"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    for asset in assets:
        asset_name = asset.get("name", "")
        if asset_name not in model_list:
            continue
        
        download_url = asset.get("browser_download_url", "")
        if not download_url:
            logger.warning(f"No download URL for {asset_name}.")
            continue
        
        target_file = output_path / asset_name
        
        # Check if file already exists
        if target_file.exists():
            logger.info(f"Skipping {asset_name} (already exists at {target_file}).")
            continue
        
        logger.info(f"Downloading {asset_name} from {tag_name}...")
        try:
            with urlopen(download_url, timeout=300) as response, open(target_file, 'wb') as out_file:
                out_file.write(response.read())
            logger.info(f"Downloaded: {target_file}")
            downloaded_count += 1
        except Exception as e:
            logger.error(f"Failed to download {asset_name}: {e}")
            # Try to clean up partial file
            if target_file.exists():
                try:
                    target_file.unlink()
                except:
                    pass
    
    if downloaded_count > 0:
        logger.info(f"Successfully downloaded {downloaded_count} model file(s).")
        return True
    else:
        logger.info("No new model files were downloaded.")
        return True


def main():
    """Main entry point for the upgrade utility."""
    parser = argparse.ArgumentParser(
        description="Upgrade BirdCrop: update ultralytics package and/or download latest YOLO models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="Only upgrade the ultralytics package, do not download models."
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only download latest models, do not upgrade the package."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to download models into (default: current directory)."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of specific models to download (e.g., 'yolo26n.pt,yolo26l.pt'). If not specified, all standard models are downloaded."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging."
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and date information for this script and the birdcrop library."
    )

    args = parser.parse_args()
    
    # --- Handle --version early ---
    if getattr(args, "version", False):
        import birdcrop
        print(f"birdcrop-upgrade command version:\t{SCRIPT_VERSION}  (date: {SCRIPT_DATE})")
        print(f"birdcrop library version:\t{birdcrop.__version__}  (date: {getattr(birdcrop, '__date__', 'unknown')})")
        sys.exit(0)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse model list if provided
    model_list = None
    if args.models:
        model_list = [m.strip() for m in args.models.split(',') if m.strip()]
        logger.info(f"Target models: {model_list}")
    
    # Decide what to upgrade
    do_package = not args.models_only
    do_models = not args.package_only
    
    package_ok = True
    models_ok = True
    
    if do_package:
        package_ok = upgrade_ultralytics_package()
    
    if do_models:
        models_ok = download_latest_models(output_dir=args.output_dir, model_list=model_list)
    
    if package_ok and models_ok:
        logger.info("Upgrade completed successfully.")
        return 0
    else:
        logger.warning("Upgrade completed with some issues (see above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
