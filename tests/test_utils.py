"""Unit tests for the shared utility helpers in utils.py.

Skipped on purpose:
    - compute_semantic_distance — requires a LangChain embeddings instance.
    - print_agent_graph — renders via Mermaid; covered manually.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import utils


# ---------------------------------------------------------------------------
# setup_logger
# ---------------------------------------------------------------------------

class TestSetupLogger:
    def test_creates_file_handler(self, tmp_path: Path):
        log_path = tmp_path / "test.log"
        logger = utils.setup_logger("test_logger_a", log_file=log_path)

        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert Path(file_handlers[0].baseFilename) == log_path

    def test_idempotent(self, tmp_path: Path):
        # Calling twice for the same logger name must not duplicate handlers.
        log_path = tmp_path / "idem.log"
        first = utils.setup_logger("test_logger_b", log_file=log_path)
        second = utils.setup_logger("test_logger_b", log_file=log_path)

        assert first is second
        assert len(first.handlers) == 1

    def test_console_handler_added_when_requested(self, tmp_path: Path):
        log_path = tmp_path / "console.log"
        logger = utils.setup_logger("test_logger_c", log_file=log_path, console=True)

        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_does_not_propagate(self, tmp_path: Path):
        log_path = tmp_path / "noprop.log"
        logger = utils.setup_logger("test_logger_d", log_file=log_path)
        assert logger.propagate is False


# ---------------------------------------------------------------------------
# load_watchlist
# ---------------------------------------------------------------------------

def _write_watchlist(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestLoadWatchlist:
    def test_loads_valid_entries(self, tmp_path: Path):
        # Pathlib joins an absolute path on the right by replacing the left,
        # so passing an absolute filename bypasses utils.py's resolve().
        path = tmp_path / "watchlist.json"
        _write_watchlist(path, {
            "competitors": [
                {"name": "Pepsi", "aliases": ["PepsiCo"]},
                {"name": "Coca-Cola", "special_focus": "EU launches"},
            ]
        })

        result = utils.load_watchlist(filename=str(path))
        assert len(result) == 2
        assert result[0].name == "Pepsi"
        assert result[0].aliases == ["PepsiCo"]
        assert result[1].name == "Coca-Cola"
        assert result[1].special_focus == "EU launches"

    def test_skips_invalid_entries(self, tmp_path: Path):
        # Missing required name field should be skipped, not crash the loader.
        path = tmp_path / "watchlist.json"
        _write_watchlist(path, {
            "competitors": [
                {"name": "Pepsi"},
                {"aliases": ["nope"]},  # invalid: no name
                {"name": "Coca-Cola"},
            ]
        })

        result = utils.load_watchlist(filename=str(path))
        assert [c.name for c in result] == ["Pepsi", "Coca-Cola"]

    def test_returns_empty_when_file_missing(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.json"
        assert utils.load_watchlist(filename=str(missing)) == []

    def test_returns_empty_when_competitors_key_absent(self, tmp_path: Path):
        path = tmp_path / "watchlist.json"
        _write_watchlist(path, {"other_key": []})
        assert utils.load_watchlist(filename=str(path)) == []

    def test_returns_empty_for_malformed_json(self, tmp_path: Path):
        path = tmp_path / "watchlist.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert utils.load_watchlist(filename=str(path)) == []
