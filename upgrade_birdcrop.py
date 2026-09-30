#!/usr/bin/env python3
"""Backward-compatible launcher for the BirdCrop upgrade utility."""

import sys

from birdcrop.upgrade import main


if __name__ == "__main__":
    sys.exit(main())
