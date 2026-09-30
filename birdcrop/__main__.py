"""Allow ``python -m birdcrop`` to invoke the BirdCrop CLI."""

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
