"""Interactive and automated execution modes for the Moodgruppen agent.

Exposes two top-level coroutines:
- run_manual_chat  — REPL loop for ad-hoc company research.
- run_automated_lookout — batch scan of every company in watchlist.json,
  with an optional follow-up Tavily deep search for significant findings.
"""

# Standard library
from datetime import date

# Local
from agent import run_agent
from config import LOGGER_NAME
from memory_sqlite3 import get_latest_company_intel
from utils import load_watchlist, setup_logger


# Shared CLI helpers.
logger = setup_logger(LOGGER_NAME)
EXIT_COMMANDS = {"exit", "quit"}


def _is_exit_command(value: str) -> bool:
    """Return True when the user entered a command to leave the workflow."""
    return value.lower() in EXIT_COMMANDS


async def run_manual_chat():
    """REPL loop for manual, interactive company research.

    Prompts the user for a company name, search engine preference, and a free-
    text research question.  Looks up the last HISTORY_LIMIT ledger entries
    for that company and injects them as PREVIOUS STATUS context before calling
    the agent.  Results are stored in SQLite after each run.
    """
    config = {"configurable": {"thread_id": "manual_user"}}
    print("\n✧ Manual Mode Ready. Type 'exit' to quit.")

    while True:
        company = input("\nEnter Company Name: ").strip()
        if _is_exit_command(company):
            break
        if not company:
            print("Company name cannot be empty.")
            continue

        # Build history context from the last N ledger entries for this company.
        # Entries are presented oldest-first so the model reads them as a timeline.
        rows = get_latest_company_intel(company)
        if rows:
            entries = "\n".join(reversed(rows))
            history_context = f"PREVIOUS STATUS (last {len(rows)} entries, oldest first):\n{entries}"
        else:
            history_context = "PREVIOUS STATUS: No prior data."

        print("Select Engine: [1] Brave Search | [2] Tavily")
        engine_choice = input("Choice: ").strip()
        if _is_exit_command(engine_choice):
            break

        user_query = input("\nYour research query: ").strip()
        if _is_exit_command(user_query):
            break
        if not user_query:
            print("Research query cannot be empty.")
            continue

        directive = ""
        if engine_choice == "1":
            engine = "brave_search"
            directive = "DIRECTIVE: Use 'brave_scout' only. "
        elif engine_choice == "2":
            engine = "tavily"
            directive = "DIRECTIVE: Use 'tavily_analyst' only. "
        else:
            print("Invalid engine selection. Please choose 1 or 2.")
            continue
            
        # Combine tool-routing directive, today's date, and user intent into a
        # lean search query.  History travels via config so sub-agents never see it.
        today = date.today().strftime("%B %d, %Y")
        query = f"{directive}TODAY: {today}. Research {company}: {user_query}."
        config["configurable"]["history_context"] = history_context

        await run_agent(query, config, company=company, engine=engine, mode="manual")


async def run_automated_lookout():
    """Scan every company in watchlist.json and flag significant changes.

    Phase 1 — Brave Scout: runs a quick broad search for each competitor and
    collects any companies whose significance score meets SIGNIFICANCE_THRESHOLD.

    Phase 2 — Tavily Deep Search (optional): for each flagged company the user
    is prompted to approve a deeper Tavily search.  A bulk "run all" option
    is available to skip per-company confirmations.
    """
    watchlist = load_watchlist()
    pending_deep_searches = []

    if not watchlist:
        print("!! No competitors found in watchlist.json")
        return

    print(f"∿ Starting automated check on {len(watchlist)} competitors...")

    # Default monitoring focus applied to every company unless overridden by
    # special_focus in watchlist.json.
    _default_focus = (
        """Monitor for new product launches in Europe or brand repositioning in Europe.
        Ignore: routine earnings reports and recycled press coverage."""
    )

    for comp in watchlist:
        company_name = comp.name
        company_focus = _default_focus
        if comp.special_focus:
            company_focus += f" Additionally: {comp.special_focus}"
        all_names = ", ".join([comp.name] + comp.aliases) if comp.aliases else comp.name

        # thread_id is passed in the config but has no effect at the moment
        # because no MemorySaver checkpointer is attached to the compiled agent.
        # It is kept here so that adding a MemorySaver later requires no changes
        # to the caller — each company would then get its own isolated thread.
        config = {"configurable": {"thread_id": f"scout_{company_name.replace(' ', '_')}"}}

        try:
            rows = get_latest_company_intel(company_name)
            if rows:
                entries = "\n".join(reversed(rows))
                history_context = f"\nPREVIOUS STATUS (last {len(rows)} entries, oldest first):\n{entries}"
            else:
                history_context = "\nPREVIOUS STATUS: No prior data."

            today = date.today().strftime("%B %d, %Y")
            query = f"DIRECTIVE: Use 'brave_scout' only. TODAY: {today}. Check {all_names}. Focus: {company_focus}."
            config["configurable"]["history_context"] = history_context

            print(f"\n⌖ [Lookout] Investigating {company_name}...")
            significant_change = await run_agent(query, config, company=company_name, engine="brave_search", mode="auto")

            if significant_change:
                print(f"☰ Queueing Deep Search for {company_name} (Significant change detected).")
                pending_deep_searches.append((company_name, config))


        except Exception as e:
            logger.error(f"Lookout failed for {company_name}: {e}")
            print(f"!! Error scouting {company_name}. Check agent.log for details.")
            continue

    # --- THE "HUMAN ARRIVAL" PHASE ---
    if not pending_deep_searches:
        print("\n✓ All checks complete. No major moves detected today.")
        return

    print("\n" + "="*50)
    print(f"DONE: Brave Search finished for all {len(watchlist)} companies.")
    print(f"Significant updates for: {', '.join([name for name, _ in pending_deep_searches])}")
    print("="*50)

    # Ask once at the start of the list
    bulk_choice = input(f"Process {len(pending_deep_searches)} deep searches? [y] Yes | [n] Skip | [a] Run All: ").strip().lower()

    if bulk_choice == 'n':
        print("Skipping all deep searches.")
        return

    for company_name, config in pending_deep_searches:
        # If user didn't choose 'Run All', ask for each one individually
        if bulk_choice != 'a':
            confirm = input(f"\n→ Start Tavily Deep Search for {company_name}? (y/n): ").strip().lower()
            if confirm != 'y':
                print(f"⏭ Skipping deep search for {company_name}.")
                continue

        # This part runs if bulk_choice was 'a' OR if individual confirm was 'y'
        print(f"∿ Running Deep Search for {company_name}...")
        today = date.today().strftime("%B %d, %Y")

        # Fetch the Brave result that was just stored so Tavily can build on it.
        brave_rows = get_latest_company_intel(company_name, limit=1)
        brave_context = (f" Brave Scout Summary:\n{brave_rows[0]}" if brave_rows else "")

        deep_query = (
            f"DIRECTIVE: Use 'tavily_analyst' only. HUMAN APPROVED DEEP SEARCH. "
            f"TODAY: {today}. The initial Brave scan for {company_name} flagged significant changes."
            f"{brave_context} "
            f"Use 'tavily_analyst' to provide a granular breakdown of initial Brave scan via primary sources."
        )
        await run_agent(deep_query, config, company=company_name, engine="tavily", mode="auto")

    print("\n✓ All deep searches processed.")
