#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.cli import validate_main  # noqa: E402

if __name__ == "__main__":
    validate_main()
