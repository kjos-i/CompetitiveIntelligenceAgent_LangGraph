"""Pytest configuration: add the evaluation folder to sys.path.

Lets test modules import eval_metrics, eval_utils, … directly
without requiring the harness to be installed as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
