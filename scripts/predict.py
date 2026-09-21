#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

script_path = Path(__file__).resolve()
candidates = [
    script_path.parents[1] / "src",
    script_path.parents[2] / "src",
    Path.cwd() / "src",
]

for src_path in candidates:
    if (src_path / "pipeline" / "cli.py").is_file():
        sys.path.insert(0, str(src_path))
        break
else:
    raise SystemExit(
        "Falta el archivo src/pipeline/cli.py en el repositorio."
    )

from pipeline.cli import predict_main  # noqa: E402

if __name__ == "__main__":
    predict_main()
