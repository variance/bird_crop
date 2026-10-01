# BirdCrop 🐦✂️

**BirdCrop** is a Python command-line utility and library designed to automatically detect objects (like birds, people, etc.) in images using YOLO models and save cropped images of those detections. It offers flexible configuration for targeting specific classes, adding margins, sorting detections, and customizing output filenames and locations using powerful templating.

Originally developed to rapidly identify and extract avian subjects from high-speed burst photography, the system has since been expanded to support multiple and diverse object classes beyond birds. It excels in its primary application: processing large volumes of in-flight bird imagery where subjects occupy only a small pixel area within the frame. By automating detection and cropping workflows for burst sequences, the tool eliminates time-intensive manual adjustments like zooming and panning while retaining analytical accuracy, making it ideal for rapid wildlife surveys and high-throughput curation of still-image datasets.

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <h3>Before (original folder)</h3>
        <img src="https://raw.githubusercontent.com/variance/bird_crop/main/docs/images/vorher.png"
     alt="image folder before auto-cropping" width="100%">
      </td>
      <td align="center" width="50%">
        <h3>After Cropping (output folder)</h3>
        <img src="https://raw.githubusercontent.com/variance/bird_crop/main/docs/images/nachher.png"
     alt="image folder after auto-cropping" width="100%">
      </td>
    </tr>
  </table>
</div>

## Key Features

*   **YOLO-Powered Detection:** Uses YOLO26 models by default for faster, more accurate bird detection. You can also provide a YOLOv8 or custom model with `--model`.
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

Install the package and its runtime dependencies from PyPI:

```bash
python -m pip install bird-crop
```

For development from a checkout:

```bash
git clone https://github.com/variance/bird_crop.git
cd bird_crop
python -m pip install .
```

This installs the `birdcrop` library and the `birdcrop` and `birdcrop-upgrade`
command-line commands. The legacy `run_birdcrop.py` and
`upgrade_birdcrop.py` launchers remain available when running from a checkout.

BirdCrop uses `yolo26n.pt` by default. If the selected model is not present in
the platform-specific BirdCrop user cache directory, the CLI downloads it
automatically from the Ultralytics assets release. Once downloaded, the model
is reused by subsequent CLI runs. You can select another YOLO26 size with `--model-size` (`nano`, `small`, `medium`,
`large`, or `xlarge`), or specify an existing YOLOv8/custom model with
`--model`.

The default model cache directory is provided by `platformdirs` and depends on
the operating system. The `birdcrop-upgrade` command uses the same directory
by default. Use `--output-dir` to download models to a different location.

### Model Selection

The default model is `yolo26n.pt`. Use `--model-size` to choose another YOLO26 model:

```bash
# default: yolo26n.pt
python run_birdcrop.py path/to/images

# use the large YOLO26 model
python run_birdcrop.py --model-size large path/to/images

# use a specific model file, including YOLOv8 or a custom model
python run_birdcrop.py --model path/to/model.pt path/to/images
```

When using `--model-size`, a missing model file is downloaded automatically. A user-specified `--model` must already exist locally.

## Usage

The installed command is `birdcrop`:

```bash
birdcrop [options] [INPUT_PATH ...]
```

The equivalent checkout command is `python run_birdcrop.py [options] [INPUT_PATH ...]`.

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

To keep BirdCrop and its YOLO models up to date, use the `birdcrop-upgrade` command:

```bash
# upgrade both the ultralytics package and download latest models
birdcrop-upgrade

# upgrade package only
birdcrop-upgrade --package-only

# download latest models only
birdcrop-upgrade --models-only

# download specific YOLO26 models
birdcrop-upgrade --models yolo26n.pt,yolo26l.pt

# download models to a specific directory instead of the default cache
birdcrop-upgrade --output-dir ./models

# verbose output
birdcrop-upgrade --verbose
```

From a checkout, `python upgrade_birdcrop.py` remains an equivalent launcher.

The upgrade utility:
- Updates `ultralytics` package via `pip install --upgrade ultralytics`
- Downloads the latest YOLO model files from the latest [ultralytics/assets](https://github.com/ultralytics/assets) release
- Skips models that already exist locally (use `--package-only` or `--models-only` to update just one component)

## Building and publishing

To build distributable artifacts locally:

```bash
python -m pip install build
python -m build
```

This creates a source distribution and wheel in `dist/`. After configuring
your PyPI credentials, upload them with:

```bash
python -m pip install twine
python -m twine upload dist/*
```
