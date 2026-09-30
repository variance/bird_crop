#!/usr/bin/env python3
"""Backward-compatible launcher for the BirdCrop command-line interface."""

import sys

from birdcrop.cli import main


if __name__ == "__main__":
    sys.exit(main())
