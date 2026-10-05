# Changelog

All notable changes to BirdCrop are documented here.

## [0.2.1] - 2026-10-05

### Added

- Added icon/logo to the top right of the README.
- Added keywords to the project description.

## [0.1.9] - 2026-10-05

### Added

- Added the pseudo object class `ALL` to select all model supported object classes
  for cropping. The combination of `--class ALL` and `--multiple` and a suitable
  `--confidence` level is potentially useful to test a model or to compare models.
- Added short options `-c` for `--classes` and `-l` for `--list-classes`.
- Added bool parameter `all_classes` to the `BirdCropper` constructor (default:
  `False`).
- Added property `target_class_names` to the BirdCropper class.

### Changed

- Improved logging to report the names of the selected classes in the summmary.
- Improved the usage message when a shortcut is invoked without drag & drop.
    (Windows)
- Changed example CLI invocations in the README to the new syntax.

## [0.1.8] - 2026-10-03

### Added

- Added automatic reuse of the best locally cached YOLO26 model when no model
  option is supplied.
- Added EXIF-aware physical image orientation before object detection.
- Added Windows desktop shortcut setup through `birdcrop-setup`, including a
  packaged BirdCrop icon and safe shortcut removal.
- Added a packaged Windows drag-and-drop launcher and documented its legacy
  status.
- Added short options `-m` for `--multiple` and `-M` for `--save-metadata`.

### Changed

- Changed the default confidence threshold used by the Windows drag-and-drop
  launcher to `0.50`.
- Changed default output-template selection so multiple target classes use
  category-specific output paths.
- Improved EXIF preservation for physically oriented crops by removing stale
  thumbnails and dimensions and resetting orientation metadata.
- Added Pillow as a direct runtime dependency for image orientation handling.
- Added release assets and packaging metadata for the Windows shortcut icon.

### Deprecated

- Deprecated direct use of the packaged `birdcrop-dragdrop.bat` launcher;
  `birdcrop-setup` now creates the preferred desktop shortcut.

### Fixed

- Fixed stale EXIF preview, dimension, autofocus-area, and orientation data in
  saved crops after image rotation.

[0.1.8]: https://github.com/variance/bird_crop/compare/v0.1.7...v0.1.8
