"""CLI launcher for the CI agent workflows."""

import asyncio

from agent_modes import run_automated_lookout, run_manual_chat


def main() -> None:
    """Entry point: prompt the user to choose a workflow and run it.

    Mode 1 — Manual: interactive REPL for ad-hoc com2pany research.
    Mode 2 — Auto:   batch watchlist scan with optional deep follow-up.
    """
    print("1: Manual Mode (Interactive Chat)")
    print("2: Auto Mode (Watchlist Automation)")
    choice = input("\nSelect Mode: ").strip()

    if choice == "1":
        asyncio.run(run_manual_chat())
    elif choice == "2":
        asyncio.run(run_automated_lookout())
    else:
        print("Invalid selection. Please choose 1 or 2.")


if __name__ == "__main__":
    main()
