# Competitive Intelligence Agent

Competitive Intelligence (CI) agent for monitoring competitors in real time. Built with LangGraph Supervisor, OpenAI GPT-4o, Brave Search, and Tavily.

In auto mode, the agent runs two specialist sub-agents under a supervisor. First, **`brave_scout`** performs a broad web scan. Then the supervisor scores each result for significance relative to previous findings, flags it if it crosses the threshold, and saves to the SQLite ledger. Once all scans are complete, a human-in-the-loop checkpoint presents the flagged findings and asks whether to proceed with more extensive searches. If approved, **`tavily_analyst`** targets the flagged changes and conducts a thorough investigation. The supervisor again scores the results and saves them to the ledger. All results are visualised in a Streamlit dashboard.

The Streamlit dashboard can be used to visualize data in the SQLite ledger at any time:

```bash
python -m streamlit run dashboard.py --server.port 8501
```

See [Run Instructions](#run-instructions) below for how to start the agent.

All scripts in this folder are Learning/Demo status.

## Use cases

This agent is built for competitive intelligence and can easily be adapted for different scenarios within that scope, such as market trend analysis, product pricing monitoring, or tracking product sentiment in social media. It can also be repurposed for any other recurring research task that benefits from scheduled web research, delta-based significance scoring, and structured reporting.

The topics searched and the metrics used to evaluate results are all configurable without touching the core agent logic. For meaningful significance scores, keep topics focused, a broad topic will almost always surface something new, which inflates scores and reduces their usefulness as a signal.

### Where to make changes for a different purpose

| What to change | Where | Effect |
|---|---|---|
| **Topic / entity list** | `watchlist.json` — add/rename entries under `"competitors"` | Controls which subjects are researched in Auto mode |
| **Search focus and scoring rules** | `system_prompt_agent.txt` — edit the ANALYSIS STANDARDS, SIGNIFICANCE SCALE, and RESPONSE FORMAT sections | Controls how the supervisor interprets results, what counts as significant, and what the report looks like |
| **Research focus (Auto mode)** | `agent_modes.py` — the `_default_focus` string in `run_automated_lookout` | The default monitoring directive sent to the supervisor for every company in the watchlist |
| **Tavily deep search prompt** | `agent_modes.py` — the `deep_query` string in `run_automated_lookout` | Controls what Tavily is asked to investigate; currently receives the Brave summary as context so it can target the specific changes Brave flagged |
| **Sub-agent search behaviour** | `agent.py` — `_brave_prompt` and `_tavily_prompt` strings | Controls what each search sub-agent looks for and prioritises |
| **Search tuning** | `config.py` — `TAVILY_TIME_RANGE`, `BRAVE_FRESHNESS`, `INCLUDED_DOMAINS`, `EXCLUDED_DOMAINS`, `SIGNIFICANCE_THRESHOLD`, `HISTORY_LIMIT`, `TEMPERATURE` | Controls recency, domain filters, what score triggers a deep search, how much history is passed as context, and how deterministic the model output is |
| **Schedule timing** | `config.py` — `SCHEDULE_INTERVAL_MINUTES` (interval mode) or `SCHEDULE_HOUR`, `SCHEDULE_MINUTE`, `SCHEDULE_DAYS` (cron mode) | Controls when `launch_schedule_runner.py` fires the automated lookout |
| **Dashboard layout and filters** | `dashboard.py` | Adapt the chart axes, sidebar filters, and displayed columns to match your topic, schedule frequency, and preferred view |
| **Database schema and stored fields** | `memory_sqlite3.py` — `CREATE TABLE` statement and `save_to_ledger` insert; `agent.py` — the `save_to_ledger(...)` call that decides which values are passed in; `memory_ledger_db.py` — column references in the CLI viewer | Change here if you want to store additional fields, rename columns, or alter what the ledger CLI displays |

## Folder Contents

### Core Python Scripts

| File | Purpose | Status |
|---|---|---|
| `agent.py` | Builds the LangGraph supervisor graph, defines both sub-agents, and exposes the `run_agent` coroutine | Learning/Demo |
| `agent_modes.py` | Two execution modes: `run_manual_chat` (interactive REPL) and `run_automated_lookout` (watchlist batch scan) | Learning/Demo |
| `launch_agent.py` | CLI entry point — prompts the user to choose Manual or Auto mode | Learning/Demo |
| `config.py` | Central control panel for all runtime flags, model settings, and search tuning | Learning/Demo |
| `memory_sqlite3.py` | SQLite persistence layer — stores and queries the intel ledger | Learning/Demo |
| `memory_ledger_db.py` | Command-line viewer for recent ledger entries, optionally filtered by company name | Learning/Demo |
| `pydantic_models.py` | Shared data models: `Company` (watchlist entry) and `SentimentResult` (structured LLM output) | Learning/Demo |
| `utils.py` | Logger setup, agent graph PNG export, watchlist loader, and semantic distance helper | Learning/Demo |
| `dashboard.py` | Streamlit dashboard for browsing and filtering saved intel results | Learning/Demo |

### Configuration and Data Files

| File | Purpose |
|---|---|
| `watchlist.json` | List of competitors to monitor in Auto mode (names, aliases, optional special focus) |
| `system_prompt_agent.txt` | Supervisor system prompt — controls routing logic, scoring rules, and output format |
| `requirements.txt` | Python dependencies |

### Runtime Artifacts

| Path | Type | Purpose |
|---|---|---|
| `agent_memory.db` | SQLite database | Intel ledger — created automatically on first run |
| `agent.log` | Log file | Runtime log created by `setup_logger()` |
| `agent_graph.png` | Image | Optional graph diagram, generated when `DRAW = True` in `config.py` |

## Architecture Overview

```
launch_agent.py
    └── agent_modes.py
            ├── run_manual_chat()       ← interactive REPL
            └── run_automated_lookout() ← watchlist batch scan
                    └── agent.py: run_agent()
                            └── LangGraph Supervisor
                                    ├── brave_scout    (Brave Search API)
                                    └── tavily_analyst (Tavily Search API)
                                            ↓
                                    memory_sqlite3.py: save_to_ledger()
                                            ↓
                                    dashboard.py (Streamlit)
```

**Flow per run (Auto mode and Scheduled mode):**

1. `agent_modes.py` looks up the last `HISTORY_LIMIT` ledger entries for the company and builds a `PREVIOUS STATUS` context block.
2. Today's date is injected into the query so the model can reject sources older than the configured cutoff.
3. The query is sent to the LangGraph supervisor. By default it routes to `brave_scout` only — the system prompt strictly forbids using `tavily_analyst` unless explicitly instructed.
4. The supervisor writes a structured report ending with `SIGNIFICANCE_SCORE: X` (1–10).
5. A second LLM call extracts a sentiment label and score.
6. Everything is saved to `agent_memory.db`.
7. After all companies are processed, any company that scored at or above `SIGNIFICANCE_THRESHOLD` is queued for a Tavily deep search. The user is prompted to approve each one (or run all at once) before `tavily_analyst` is invoked.

## Scoring System

Significance is scored **relative to PREVIOUS STATUS** — not by the absolute importance of the news:

| Score | Meaning |
|---|---|
| 1–2 | No meaningful new information vs. previous run |
| 3–4 | Minor new detail or clarification on an already-known story |
| 5–6 | Genuinely new incremental development |
| 7–8 | New strategic shift (pricing, product, positioning, partnership) |
| 9–10 | Major new move with strong commercial or competitive impact |

Scores ≥ `SIGNIFICANCE_THRESHOLD` (default: 7) trigger a Tavily deep search in Auto mode.

## Configuration

All tuning is done in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `MODEL_NAME` | `"gpt-4o"` | OpenAI model used by all agents |
| `TEMPERATURE` | `0.0` | Low temperature for factual, deterministic output |
| `SIGNIFICANCE_THRESHOLD` | `7` | Minimum score to trigger a deep search / alert |
| `HISTORY_LIMIT` | `5` | Number of previous ledger entries passed as context |
| `TAVILY_MAX_RESULTS` | `3` | Results per Tavily call |
| `TAVILY_SEARCH_DEPTH` | `"advanced"` | `"basic"` or `"advanced"` |
| `TAVILY_TIME_RANGE` | `"week"` | Restrict Tavily results to the past week |
| `BRAVE_MAX_RESULTS` | `5` | Results per Brave Search call |
| `BRAVE_FRESHNESS` | `"pw"` | `"pd"` = past day, `"pw"` = past week, `"pm"` = past month |
| `EXCLUDED_DOMAINS` | `["linkedin.com"]` | Domains blocked from all results |
| `INCLUDED_DOMAINS` | `[]` | Domains to prioritise (empty = no preference) |
| `SCHEDULE_HOUR` | `9` | Cron schedule: hour to run the lookout (24h format) — used when `SCHEDULE_INTERVAL_MINUTES` is `0` |
| `SCHEDULE_MINUTE` | `0` | Cron schedule: minute to run the lookout — used when `SCHEDULE_INTERVAL_MINUTES` is `0` |
| `SCHEDULE_DAYS` | `"mon-fri"` | Cron schedule: days to run (`"mon-fri"`, `"mon-sun"`, etc.) — used when `SCHEDULE_INTERVAL_MINUTES` is `0` |
| `SCHEDULE_INTERVAL_MINUTES` | `1` | Set to a positive integer to run every N minutes (interval mode). Set to `0` to use the cron schedule above instead |
| `UPDATES` | `True` | Print active node names during execution |
| `EVENTS` | `False` | Print every raw LangGraph event (verbose) |
| `MESSAGES` | `False` | Print full message history at end of stream |
| `DRAW` | `False` | Save a PNG of the agent graph on startup |

## Watchlist (`watchlist.json`)

Add competitors to monitor in Auto mode:

```json
{
  "competitors": [
    {
      "name": "Pepsi",
      "aliases": ["PepsiCo", "Pepsi-Cola"],
      "special_focus": ""
    },
    {
      "name": "Coca-Cola",
      "aliases": ["Coke", "The Coca-Cola Company"],
      "special_focus": ""
    }
  ]
}
```

- **`name`** — canonical name used as the database key
- **`aliases`** — alternative names included in search queries for broader coverage
- **`special_focus`** — optional extra instruction appended to the default search directive

## Requirements

```
altair
apscheduler
langchain
langchain-community
langchain-core
langchain-openai
langchain-tavily
langgraph-supervisor
numpy
pandas
pydantic
python-dotenv
streamlit
streamlit-autorefresh
```

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Environment Variables

Two API keys are required:

```env
OPENAI_API_KEY=...
BRAVE_SEARCH_API_KEY=...
TAVILY_API_KEY=...
```

The `.env` file path is set directly in `agent.py`:

```python
load_dotenv()
```

Update this path to match your local setup before running.

## Run Instructions

### Scheduled Automation (primary / recommended)

`launch_schedule_runner.py` is the main way to run the agent in production. It uses APScheduler to run `run_automated_lookout` on a recurring schedule. Two modes are supported, controlled by `SCHEDULE_INTERVAL_MINUTES` in `config.py`:

- **Interval mode** (`SCHEDULE_INTERVAL_MINUTES > 0`): runs every N minutes. Set to `1` to run every minute.
- **Cron mode** (`SCHEDULE_INTERVAL_MINUTES = 0`): runs at a fixed time of day, configured via `SCHEDULE_HOUR`, `SCHEDULE_MINUTE`, and `SCHEDULE_DAYS` (default: weekdays at 09:00).

```bash
python launch_schedule_runner.py
```

Keep the process running in a terminal or deploy it via a process manager (systemd, supervisor, etc.). Press `Ctrl+C` to stop. The schedule and timing are configured in `config.py` (`SCHEDULE_HOUR`, `SCHEDULE_MINUTE`, `SCHEDULE_DAYS`).

### Agent (Manual or Auto mode — interactive / one-off)

```bash
python launch_agent.py
```

Select mode at the prompt:

```
1: Manual Mode (Interactive Chat)
2: Auto Mode (Watchlist Automation)
```

**Manual mode** — enter a company name, choose a search engine, and type a free-text research question.

**Auto mode** — scans all companies in `watchlist.json` with Brave Search, then optionally runs a Tavily deep search for any company that scored above the significance threshold.

### Streamlit Dashboard

Run from within the `CompetitiveIntelligenceAgent_LangGraph` folder:

```bash
python -m streamlit run dashboard.py --server.port 8501
```

The dashboard polls the database every 10 seconds and refreshes automatically when new rows are detected.

## Suggested Learning Path

1. Start with `config.py` to understand all tunable settings.
2. Read `system_prompt_agent.txt` to see how the supervisor makes routing and scoring decisions.
3. Read `agent.py` to understand how the supervisor graph and sub-agents are constructed.
4. Read `agent_modes.py` to see how queries are assembled (date injection, history context, directives).
5. Read `memory_sqlite3.py` to understand the persistence layer.
6. Run `launch_agent.py` in Manual mode for a live test.
7. Open `dashboard.py` to explore results visually.
