# BirdCrop 🐦✂️

**BirdCrop** is a Python command-line utility and library designed to automatically detect objects (like birds, people, etc.) in images using YOLO models and save cropped images of those detections. It offers flexible configuration for targeting specific classes, adding margins, sorting detections, and customizing output filenames and locations using powerful templating.

Originally developed to rapidly identify and extract avian subjects from high-speed burst photography, the system has since been expanded to support multiple and diverse object classes beyond birds. It excels in its primary application: processing large volumes of in-flight bird imagery where subjects occupy only a small pixel area within the frame. By automating detection and cropping workflows for burst sequences, the tool eliminates time-intensive manual adjustments like zooming and panning while retaining analytical accuracy, making it ideal for rapid wildlife surveys and high-throughput curation of still-image datasets.

## Key Features

*   **YOLO-Powered Detection:** Leverages Ultralytics YOLO models (like YOLOv8) for object detection. Use pre-trained models or your custom ones.
*   **Flexible Class Targeting:** Specify which object classes to detect using their names (e.g., `"bird,dog,cat"`) or their model-specific IDs (e.g., `"14,16,15"`). The tool adapts to the classes present in the loaded model.
*   **List Model Classes:** Easily list all classes and their IDs available within a specific YOLO model file using the `--list-classes` option.
*   **Customizable Output Paths:** Define complex output file paths and names using Python's format string syntax via `--output-template`. Access detailed information about the input file, detection, and crop (see Template Variables below).
*   **Cropping Margin:** Add a specified pixel margin around the detected bounding box before cropping using `--margin`.
*   **Detection Sorting:** Sort multiple detections within an image by `confidence` or bounding box `size` (default) using `--sortby`.
*   **Single or Multiple Crops:** Choose to save only the single "best" detection (highest confidence or largest size) per image or save crops for *all* detected objects using `--multiple`.
*   **Per-Category Numbering:** Use the `{pcnr}` template variable for sequential numbering *within* each category for a given input image.
*   **Directory Processing:** Process all supported images within specified directories, optionally searching recursively (`-r`).
*   **Concurrent Processing:** Speed up processing on multi-core systems using parallel worker threads (`-w`).
*   **Overwrite Control:** Prevent accidental data loss by default; use `--force` (`-f`) to allow overwriting existing crop files.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/variance/bird_crop.git
    cd bird_crop
    ```

2.  **Install dependencies:**
    BirdCrop relies on several Python packages. You can install them using pip:
    ```bash
    pip install ultralytics opencv-python numpy piexif
    ```
    *   `ultralytics`: For the YOLO model loading and prediction.
    *   `opencv-python`: For image reading, writing, and cropping.
    *   `numpy`: For numerical operations.
    *   `piexif`: For reading EXIF metadata from images (used in templating).

3.  **(Optional) Download a YOLO model:** If you don't have one, download a pre-trained model (like `yolov8n.pt` used by default) from the Ultralytics YOLO releases. Place it where the script can find it (e.g., in the `bird_crop` directory) or specify the path using `--model`.

## Usage

The main script is `run_birdcrop.py`.

```bash
python run_birdcrop.py [options] [INPUT_PATH ...]
```

## Update Checks

BirdCrop can check for upstream updates at startup.

- Package versions: https://pypi.org/pypi/ultralytics/json
- YOLO asset release tags: https://api.github.com/repos/ultralytics/assets/releases/latest

CLI flags:

```bash
# enabled by default
python run_birdcrop.py --check-updates [options] [INPUT_PATH ...]

# disable all online checks
python run_birdcrop.py --no-update-check [options] [INPUT_PATH ...]

# network timeout per endpoint in seconds
python run_birdcrop.py --update-check-timeout 2.0 [options] [INPUT_PATH ...]
```

## Upgrading

To keep BirdCrop and its YOLO models up to date, use the `upgrade_birdcrop.py` utility:

```bash
# upgrade both package and download latest models
python upgrade_birdcrop.py

# upgrade package only
python upgrade_birdcrop.py --package-only

# download latest models only
python upgrade_birdcrop.py --models-only

# download specific models
python upgrade_birdcrop.py --models yolov8n.pt,yolov8l.pt

# download models to a specific directory
python upgrade_birdcrop.py --output-dir ./models

# verbose output
python upgrade_birdcrop.py --verbose
```

The upgrade utility:
- Updates `ultralytics` package via `pip install --upgrade ultralytics`
- Downloads the latest YOLO model files from the latest [ultralytics/assets](https://github.com/ultralytics/assets) release
- Skips models that already exist locally (use `--package-only` or `--models-only` to update just one component)
