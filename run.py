#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure repository root and src directory are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"

for _path in [str(_REPO_ROOT), str(_SRC_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.run.main import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
