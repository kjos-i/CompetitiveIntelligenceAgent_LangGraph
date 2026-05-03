"""pytest configuration for the parent-project test suite.

Adds the project root to sys.path so test modules can import
top-level modules (memory_sqlite3, utils, pydantic_models,
…) without packaging the project. Mirrors the same pattern used by
evaluation/tests/conftest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
