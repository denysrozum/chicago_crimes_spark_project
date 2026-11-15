"""Package entry point for `python -m chicago_crimes`."""

import sys

from chicago_crimes.cli import main

if __name__ == "__main__":
    sys.exit(main())
