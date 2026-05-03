"""SQLite persistence layer for the CI agent.

Provides three operations on the intel_ledger table:
- init_db                  — create the table on first run.
- get_latest_company_intel — fetch the last N reports for a company.
- save_to_ledger           — persist a completed research report.

The database file is stored next to this module as agent_memory.db.
"""

# Standard library
import sqlite3
from datetime import datetime
from pathlib import Path

# Local
from config import HISTORY_LIMIT


DB_PATH = Path(__file__).resolve().parent / "agent_memory.db"


def init_db() -> None:
    """Create the intelligence ledger table if it does not already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS intel_ledger 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        Company TEXT,
                        Query TEXT, 
                        Result TEXT, 
                        Significance INTEGER,
                        Importance INTEGER DEFAULT 0,
                        Description TEXT,
                        Sentiment INTEGER,
                        Engine TEXT,
                        Mode TEXT,   
                        Timestamp DATETIME)''')
        conn.commit()


def get_latest_company_intel(company_name: str, limit: int = HISTORY_LIMIT) -> list[str]:
    """Return the most recent ledger results for *company_name*.

    Rows are ordered newest-first and capped at *limit* (default:
    HISTORY_LIMIT from config.py).
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Result FROM intel_ledger WHERE Company = ? ORDER BY Timestamp DESC LIMIT ?",
            (company_name, limit),
        )
        rows = cursor.fetchall()
    return [row[0] for row in rows]  # list of result strings, newest first


def save_to_ledger(
    company: str,
    query: str,
    result: str,
    significance_score: int,
    significance_flag: bool = False,
    sentiment: str | None = None,
    sentiment_score: int | None = None,
    engine: str | None = None,
    mode: str | None = None,
) -> None:
    """Persist a completed research report to the intel ledger.

    Column mapping:
    - Description stores the one-word sentiment label (e.g. "positive").
    - Sentiment   stores the numeric sentiment score (1–10).
    - Importance  is set to 1 when *significance_flag* is True (i.e. the
      score met or exceeded SIGNIFICANCE_THRESHOLD).
    """
    query_sql = """
        INSERT INTO intel_ledger 
        (Company, Query, Result, Significance, Importance, Description, Sentiment, Engine, Mode, Timestamp) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = (
        company,
        query,
        result,
        significance_score,
        1 if significance_flag else 0,
        sentiment,
        sentiment_score,
        engine,
        mode,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query_sql, values)
        conn.commit()
