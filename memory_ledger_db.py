"""Command-line viewer for recent entries in the intelligence ledger."""

import sqlite3
import sys
from pathlib import Path

import pandas as pd


DEFAULT_COMPANY = ""
DEFAULT_LIMIT = 2

DB_PATH = Path(__file__).resolve().parent / "agent_memory.db"


def view_ledger(company_name=DEFAULT_COMPANY, limit=DEFAULT_LIMIT):
    """View recent ledger entries, optionally filtered by company name."""
    if not DB_PATH.exists():
        print(f"!! Error: Database file not found at {DB_PATH}")
        return

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        print("!! Error: limit must be an integer.")
        return

    if limit <= 0:
        print("!! Error: limit must be greater than 0.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)

        if company_name:
            query = """
                SELECT * FROM intel_ledger
                WHERE lower(company) = lower(?)
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(company_name, limit))
            title = f"STRATEGIC INTELLIGENCE LEDGER — {company_name}"
        else:
            query = """
                SELECT * FROM intel_ledger
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
            title = "STRATEGIC INTELLIGENCE LEDGER — ALL COMPANIES"

        conn.close()

        if df.empty:
            if company_name:
                print(f"▤ No ledger entries found for '{company_name}'.")
            else:
                print("▤ Ledger is currently empty.")
            return

        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

        for _, row in df.iterrows():
            print(f"\n[ ENTRY #{row['id']} ]")
            print(f"◈ Company: {row['Company']}")
            print(f"  Significance: {row['Significance']}")
            print(f"  Importance: {'★' if row['Importance'] == 1 else '○'}")
            print(f"  Description: {row.get('Description', 'n/a')}")
            print(f"  Sentiment: {row['Sentiment']}")
            print(f"  Engine: {row['Engine']}")
            print(f"  Mode: {row['Mode']}")
            print(f"  Timestamp: {row['Timestamp']}")
            print(f"  Query: {row['Query']}")
            print(f"\nResult:\n\n{row['Result']}")
            print("\n" + "*" * 80 + "\n")

    except Exception as e:
        print(f"!! Error reading database: {e}")


if __name__ == "__main__":
    company_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COMPANY
    limit_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_LIMIT
    view_ledger(company_name=company_arg, limit=limit_arg)
