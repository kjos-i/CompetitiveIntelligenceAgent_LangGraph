"""Unit tests for the SQLite persistence layer.

Each test gets an isolated database file under tmp_path by
monkeypatching memory_sqlite3.DB_PATH — the real
agent_memory.db is never touched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import memory_sqlite3


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point memory_sqlite3 at a fresh DB inside tmp_path and init it."""
    db_path = tmp_path / "test_intel.db"
    monkeypatch.setattr(memory_sqlite3, "DB_PATH", db_path)
    memory_sqlite3.init_db()
    return db_path


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_intel_ledger_table(self, temp_db: Path):
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='intel_ledger'"
            ).fetchone()
        assert row is not None and row[0] == "intel_ledger"

    def test_idempotent(self, temp_db: Path):
        # Calling init_db twice must not raise.
        memory_sqlite3.init_db()
        memory_sqlite3.init_db()


# ---------------------------------------------------------------------------
# save_to_ledger
# ---------------------------------------------------------------------------

class TestSaveToLedger:
    def test_basic_save(self, temp_db: Path):
        memory_sqlite3.save_to_ledger(
            company="Pepsi",
            query="test query",
            result="test result",
            significance_score=5,
        )
        with sqlite3.connect(temp_db) as conn:
            rows = conn.execute(
                "SELECT Company, Query, Result, Significance FROM intel_ledger"
            ).fetchall()
        assert rows == [("Pepsi", "test query", "test result", 5)]

    def test_significance_flag_translated_to_importance_int(self, temp_db: Path):
        memory_sqlite3.save_to_ledger(
            "Pepsi", "q", "r", 8, significance_flag=True,
        )
        memory_sqlite3.save_to_ledger(
            "Coca-Cola", "q", "r", 3, significance_flag=False,
        )
        with sqlite3.connect(temp_db) as conn:
            rows = conn.execute(
                "SELECT Company, Importance FROM intel_ledger ORDER BY id"
            ).fetchall()
        assert rows == [("Pepsi", 1), ("Coca-Cola", 0)]

    def test_optional_fields_nullable(self, temp_db: Path):
        # sentiment / sentiment_score / engine / mode default to None.
        memory_sqlite3.save_to_ledger("Pepsi", "q", "r", 5)
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT Description, Sentiment, Engine, Mode FROM intel_ledger"
            ).fetchone()
        assert row == (None, None, None, None)

    def test_persists_sentiment_and_engine_metadata(self, temp_db: Path):
        memory_sqlite3.save_to_ledger(
            "Pepsi", "q", "r", 7,
            sentiment="positive", sentiment_score=8,
            engine="brave_search", mode="auto",
        )
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT Description, Sentiment, Engine, Mode FROM intel_ledger"
            ).fetchone()
        assert row == ("positive", 8, "brave_search", "auto")


# ---------------------------------------------------------------------------
# get_latest_company_intel
# ---------------------------------------------------------------------------

class TestGetLatestCompanyIntel:
    def test_returns_results_newest_first(self, temp_db: Path):
        # Insert three rows with explicit timestamps so ordering is deterministic.
        with sqlite3.connect(temp_db) as conn:
            conn.executemany(
                "INSERT INTO intel_ledger (Company, Result, Timestamp) VALUES (?, ?, ?)",
                [
                    ("Pepsi", "old",   "2026-01-01 09:00:00"),
                    ("Pepsi", "newer", "2026-02-01 09:00:00"),
                    ("Pepsi", "newest","2026-03-01 09:00:00"),
                ],
            )
            conn.commit()

        results = memory_sqlite3.get_latest_company_intel("Pepsi", limit=10)
        assert results == ["newest", "newer", "old"]

    def test_respects_limit(self, temp_db: Path):
        with sqlite3.connect(temp_db) as conn:
            conn.executemany(
                "INSERT INTO intel_ledger (Company, Result, Timestamp) VALUES (?, ?, ?)",
                [
                    ("Pepsi", "r1", "2026-01-01 09:00:00"),
                    ("Pepsi", "r2", "2026-02-01 09:00:00"),
                    ("Pepsi", "r3", "2026-03-01 09:00:00"),
                ],
            )
            conn.commit()

        results = memory_sqlite3.get_latest_company_intel("Pepsi", limit=2)
        assert len(results) == 2
        assert results[0] == "r3"  # newest first

    def test_filters_by_company(self, temp_db: Path):
        memory_sqlite3.save_to_ledger("Pepsi", "q", "pepsi-result", 5)
        memory_sqlite3.save_to_ledger("Coca-Cola", "q", "coke-result", 5)

        assert memory_sqlite3.get_latest_company_intel("Pepsi") == ["pepsi-result"]
        assert memory_sqlite3.get_latest_company_intel("Coca-Cola") == ["coke-result"]

    def test_returns_empty_for_unknown_company(self, temp_db: Path):
        assert memory_sqlite3.get_latest_company_intel("UnknownCo") == []
