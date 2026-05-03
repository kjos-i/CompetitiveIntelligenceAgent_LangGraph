"""SQLite ledger for persisting evaluation results across runs.

Creates and manages eval_ledger.db in the evaluation folder with two tables:
  - eval_runs:  one row per evaluation run (summary-level metrics).
  - eval_cases: one row per case per run (per-case metrics and details).

Schema is generated automatically from the metric registry, so adding a metric
to eval_metric_registry.py automatically extends both tables on next startup
via the auto-migration block in _ensure_tables().

Schema policy
-------------
The auto-migration is **add-only**. _try_add_column will append new
columns when a metric is added to the registry, but it does not rename
or drop columns when one is removed or renamed. By design — the harness
does not maintain compatibility shims for old column names or manual
ALTER TABLE scripts.

When a registry change renames or removes a metric column, the
recommended fix is to save the existing eval_ledger.db aside under
a new name before the next run, so historical data is preserved under
its original schema. If the current data is disposable, the file can
simply be removed. To save aside or remove::

    mv evaluation/eval_ledger.db evaluation/eval_ledger_<timestamp>.db
    # or, if disposable:
    rm evaluation/eval_ledger.db
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from eval_metric_registry import case_sql_columns, judge_metric_keys, run_sql_columns
from eval_config import FULL_METRIC_GROUPS


def _try_add_column(conn: sqlite3.Connection, table: str, col_def: str) -> None:
    """ALTER TABLE ... ADD COLUMN, swallowing only the 'already exists' error.

    SQLite reports duplicate columns as OperationalError with a message
    starting "duplicate column name". Any other OperationalError
    (malformed SQL, locked DB, disk full, permission issue) is re-raised so
    a real schema problem can't silently corrupt later writes — the traceback
    reaches stderr via Python's default exception handler.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return
        raise

DB_PATH = Path(__file__).resolve().parent / "eval_ledger.db"

# ---------------------------------------------------------------------------
# Schema — generated from the metric registry
# ---------------------------------------------------------------------------

_RUNS_FIXED_PREFIX = """\
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT,
    layer               INTEGER DEFAULT 2,
    run_timestamp       TEXT    NOT NULL,
    dataset_path        TEXT,
    judge_model         TEXT,
    threshold           REAL,
    execution_mode      TEXT,
    case_count          INTEGER,
    pass_count          INTEGER,
    pass_rate           REAL,
    avg_judge_run_score REAL,"""

_RUNS_FIXED_SUFFIX = """\
    enabled_groups      TEXT,
    gate_thresholds     TEXT,
    metric_averages     TEXT
);"""

_CASES_FIXED_PREFIX = """\
CREATE TABLE IF NOT EXISTS eval_cases (
    case_row_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER NOT NULL REFERENCES eval_runs(run_id),
    run_timestamp       TEXT    NOT NULL,
    case_id             TEXT    NOT NULL,
    layer               INTEGER NOT NULL,
    category            TEXT    DEFAULT '',
    question            TEXT,
    status              TEXT,"""

_CASES_FIXED_SUFFIX = """\
    error_count         INTEGER,
    answer              TEXT,
    expected_output     TEXT,
    errors              TEXT,
    failure_reasons     TEXT
);"""


def _build_create_sql(prefix: str, columns: list[tuple[str, str]], suffix: str) -> str:
    column_lines = "\n".join(f"    {name:<36s}{datatype}," for name, datatype in columns)
    return f"{prefix}\n{column_lines}\n{suffix}"


_CREATE_RUNS_TABLE  = _build_create_sql(_RUNS_FIXED_PREFIX,  run_sql_columns(),  _RUNS_FIXED_SUFFIX)
_CREATE_CASES_TABLE = _build_create_sql(_CASES_FIXED_PREFIX, case_sql_columns(), _CASES_FIXED_SUFFIX)


# ---------------------------------------------------------------------------
# Ledger class
# ---------------------------------------------------------------------------

class EvalLedger:
    """Thin wrapper around a SQLite database that stores evaluation runs and per-case results."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self) -> None:
        """Create the schema if absent and add any newly registered metric columns.

        The CREATE TABLE statements always run with IF NOT EXISTS, so
        first-time setup just creates the tables. The _try_add_column
        loops only matter when a metric was added to eval_metric_registry
        after the ledger DB was created — they extend an existing ledger to
        include the new columns without requiring a wipe.
        """
        with self._connect() as conn:
            conn.execute(_CREATE_RUNS_TABLE)
            conn.execute(_CREATE_CASES_TABLE)
            for col, typ in case_sql_columns():
                _try_add_column(conn, "eval_cases", f"{col} {typ}")
            for col, typ in run_sql_columns():
                _try_add_column(conn, "eval_runs", f"{col} {typ}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_run(self, report: dict[str, Any], execution_mode: str = "fixture", session_id: str | None = None) -> int | None:
        """Persist a full evaluation report (summary + per-case results).

        Only runs where all three metric groups (judge, format, routing) were
        enabled are written to the ledger so trend charts compare
        like-with-like. Partial runs still produce JSON/CSV artifacts but
        are not stored here.
        """
        summary = report.get("summary", {})
        enabled_groups_raw = set(summary.get("enabled_groups") or [])
        if enabled_groups_raw != FULL_METRIC_GROUPS:
            return None

        run_avg_cols = [col for col, _ in run_sql_columns()]
        run_values = (
            session_id,
            report.get("layer", 2),
            report.get("generated_at"),
            report.get("dataset_path"),
            report.get("judge_model"),
            report.get("threshold"),
            execution_mode,
            summary.get("case_count"),
            summary.get("pass_count"),
            summary.get("pass_rate"),
            summary.get("avg_judge_run_score"),
            *[summary.get(col) for col in run_avg_cols],
            json.dumps(summary.get("enabled_groups")) if summary.get("enabled_groups") is not None else None,
            json.dumps(report.get("gate_thresholds")) if report.get("gate_thresholds") is not None else None,
            json.dumps(summary.get("metric_averages", {})),
        )

        with self._connect() as conn:
            run_id = _insert_run_row(conn, run_values)
            _insert_case_rows(
                conn,
                run_id=run_id,
                run_timestamp=report.get("generated_at"),
                results=report.get("results", []),
                default_layer=2,
            )

        return run_id

    def save_l1_run(self, report: dict[str, Any], session_id: str | None = None) -> int:
        """Persist a Layer 1 sub-agent isolation run to the ledger.

        Unlike save_run(), there is no FULL_METRIC_GROUPS guard — every
        Layer 1 run is recorded because it tests the live API integration
        and all its metrics are always-on by definition.
        """
        summary = report.get("summary", {})
        results = report.get("results", [])

        run_avg_cols = [col for col, _ in run_sql_columns()]
        run_values = (
            session_id,
            1,                              # layer
            report.get("generated_at"),
            report.get("dataset_path"),
            None,                           # judge_model — not applicable for L1
            None,                           # threshold   — not applicable for L1
            "live",                         # always live for L1
            summary.get("case_count"),
            summary.get("pass_count"),
            summary.get("pass_rate"),
            None,                           # avg_judge_run_score — no judge score for L1
            # Pull every registry-derived average straight from the summary.
            # L1 doesn't run judge / format / routing metrics, so those
            # columns are None automatically; sub-agent metrics
            # (result_count, domain_*) populate via the same loop.
            *[summary.get(col) for col in run_avg_cols],
            json.dumps([]),                 # enabled_groups
            None,                           # gate_thresholds
            json.dumps(summary.get("metric_averages", {})),
        )

        with self._connect() as conn:
            run_id = _insert_run_row(conn, run_values)
            _insert_case_rows(
                conn,
                run_id=run_id,
                run_timestamp=report.get("generated_at"),
                results=results,
                default_layer=1,
            )

        return run_id

    def get_recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent evaluation run summaries."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM eval_runs ORDER BY run_id DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_cases_for_run(self, run_id: int) -> list[dict[str, Any]]:
        """Return all per-case rows for a given run."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM eval_cases WHERE run_id = ? ORDER BY case_row_id", (run_id,)
            )
            return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RUN_FIXED_COLS = [
    "session_id", "layer", "run_timestamp", "dataset_path", "judge_model", "threshold",
    "execution_mode", "case_count", "pass_count", "pass_rate", "avg_judge_run_score",
]
_RUN_TRAILING_COLS = ["enabled_groups", "gate_thresholds", "metric_averages"]

_CASE_FIXED_COLS = [
    "run_id", "run_timestamp", "case_id", "layer",
    "category", "question", "status",
]
_CASE_TRAILING_COLS = [
    "error_count",
    "answer", "expected_output",
    "errors", "failure_reasons",
]


def _columns_in_sql_block(sql: str) -> list[str]:
    """Extract column names declared in a CREATE TABLE SQL block.

    Used at import time to verify that the hardcoded _RUN_FIXED_COLS
    / _CASE_FIXED_COLS / *_TRAILING_COLS lists match the column
    names embedded in the SQL templates above. Without this check, a
    schema edit that touched only the SQL string would silently misalign
    every INSERT by one column.
    """
    columns: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped or stripped.startswith(("CREATE", ")", "--")):
            continue
        token = stripped.split(None, 1)[0]
        # Skip table-level constraints (PRIMARY KEY, FOREIGN KEY, etc.).
        if token.isupper():
            continue
        columns.append(token)
    return columns


def _assert_fixed_cols_align_with_sql() -> None:
    """Fail loudly at import time if the SQL templates and column lists drift."""
    runs_declared = _columns_in_sql_block(_RUNS_FIXED_PREFIX) + _columns_in_sql_block(_RUNS_FIXED_SUFFIX)
    expected_runs = ["run_id"] + _RUN_FIXED_COLS + _RUN_TRAILING_COLS
    assert runs_declared == expected_runs, (
        "eval_runs SQL prefix/suffix and _RUN_FIXED_COLS/_RUN_TRAILING_COLS drifted.\n"
        f"  in SQL: {runs_declared}\n  in lists: {expected_runs}"
    )

    cases_declared = _columns_in_sql_block(_CASES_FIXED_PREFIX) + _columns_in_sql_block(_CASES_FIXED_SUFFIX)
    expected_cases = ["case_row_id"] + _CASE_FIXED_COLS + _CASE_TRAILING_COLS
    assert cases_declared == expected_cases, (
        "eval_cases SQL prefix/suffix and _CASE_FIXED_COLS/_CASE_TRAILING_COLS drifted.\n"
        f"  in SQL: {cases_declared}\n  in lists: {expected_cases}"
    )


_assert_fixed_cols_align_with_sql()


def _run_columns() -> list[str]:
    """Ordered column list for an eval_runs INSERT.

    Built from the registry-derived run averages so adding a metric with a
    summary_avg_key automatically extends both the schema and the INSERT
    in lockstep.
    """
    return _RUN_FIXED_COLS + [col for col, _ in run_sql_columns()] + _RUN_TRAILING_COLS


def _case_columns() -> list[str]:
    """Ordered column list for an eval_cases INSERT."""
    return _CASE_FIXED_COLS + [col for col, _ in case_sql_columns()] + _CASE_TRAILING_COLS


def _insert_run_row(conn: sqlite3.Connection, run_values: tuple[Any, ...]) -> int:
    """Insert a single eval_runs row and return the new run_id."""
    cols = _run_columns()
    placeholders = ", ".join("?" for _ in cols)
    cursor = conn.execute(
        f"INSERT INTO eval_runs ({', '.join(cols)}) VALUES ({placeholders})",
        run_values,
    )
    return cursor.lastrowid  # type: ignore[return-value]


def _insert_case_rows(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    run_timestamp: str | None,
    results: list[dict[str, Any]],
    default_layer: int,
) -> None:
    """Insert one eval_cases row per result.

    Shared between save_run and save_l1_run so both layers persist
    the same trailing diagnostics columns (errors, failure_reasons)
    and benefit from any future schema additions in one place.
    """
    cols = _case_columns()
    placeholders = ", ".join("?" for _ in cols)
    judge_keys = judge_metric_keys()

    for item in results:
        metrics = item.get("judge_metrics", {})
        kw      = item.get("keyword_checks", {})
        errors  = item.get("errors", [])
        metric_values = _extract_case_metric_values(item, metrics, kw, judge_keys)

        case_values = (
            run_id,
            run_timestamp,
            item.get("id"),
            item.get("layer", default_layer),
            item.get("category", ""),
            item.get("question"),
            item.get("status"),
            *metric_values,
            len(errors),
            item.get("answer"),
            item.get("expected_output"),
            json.dumps(errors),
            json.dumps(item.get("failure_reasons", [])),
        )
        conn.execute(
            f"INSERT INTO eval_cases ({', '.join(cols)}) VALUES ({placeholders})",
            case_values,
        )


def _extract_case_metric_values(
    item: dict[str, Any],
    metrics: dict[str, Any],
    kw: dict[str, Any],
    judge_keys: list[str],
) -> tuple[Any, ...]:
    """Extract metric values in the same order as case_sql_columns()."""
    values: list[Any] = []
    for col, _ in case_sql_columns():
        if col in judge_keys:
            values.append(metrics.get(col, {}).get("score"))
        elif col == "required_keyword_hit_rate":
            values.append(kw.get("required_keyword_hit_rate"))
        elif col == "disallowed_keyword_hits":
            values.append(kw.get("disallowed_keyword_hits"))
        elif col == "avg_judge_score":
            values.append(item.get("avg_judge_score"))
        else:
            values.append(item.get(col))
    return tuple(values)
