#!/usr/bin/env python3
"""rc2 entry point for dnn."""

from __future__ import annotations

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from model import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
