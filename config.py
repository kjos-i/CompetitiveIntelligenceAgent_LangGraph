"""
Global configuration settings for the CI agent.

All runtime behaviour, model/tool tuning, and debug flags are controlled
from this single file.  Import the required constants in other modules rather
than hard-coding values there.
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Logger name shared across all modules
LOGGER_NAME = "live_CI_agent"

# ---------------------------------------------------------------------------
# Debug / verbosity flags
# ---------------------------------------------------------------------------
# Print active node names while the graph is running
UPDATES = True

# Print every raw LangGraph event (very verbose — useful for debugging)
EVENTS = False

# Print the full message history at the end of each stream
MESSAGES = False

# Save a PNG diagram of the compiled agent graph on startup
DRAW = False

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0          # Low temperature for factual, deterministic output
SIGNIFICANCE_THRESHOLD = 7 # Minimum score to trigger a deep search / alert
HISTORY_LIMIT = 7          # Previous ledger entries sent to the agent as context

# ---------------------------------------------------------------------------
# Tavily search settings
# ---------------------------------------------------------------------------
TAVILY_MAX_RESULTS = 3          # Number of results returned per search call
TAVILY_SEARCH_DEPTH = "advanced" # "basic" is faster; "advanced" is more thorough
TAVILY_TOPIC = "general"        # "general" | "news" | "finance"
TAVILY_TIME_RANGE = "week"      # Restrict results to the past week

# ---------------------------------------------------------------------------
# Brave Search settings
# ---------------------------------------------------------------------------
BRAVE_MAX_RESULTS = 5       # Number of results returned per search call
BRAVE_FRESHNESS = "pw"      # "pd" = past day, "pw" = past week, "pm" = past month
BRAVE_EXTRA_SNIPPETS = True # Include extended result snippets when available

# ---------------------------------------------------------------------------
# Domain allow/block lists (applied to both Tavily natively and Brave via prompt)
# ---------------------------------------------------------------------------
# Sources to exclude from all search results
EXCLUDED_DOMAINS = [
    "linkedin.com",
]

# Sources to prioritise in search results (empty = no preference)
INCLUDED_DOMAINS = []

# ---------------------------------------------------------------------------
# Schedule settings (used by launch_schedule_runner.py)
# ---------------------------------------------------------------------------
SCHEDULE_HOUR = 9           # Hour to run the automated lookout (24h format)
SCHEDULE_MINUTE = 0         # Minute to run the automated lookout
SCHEDULE_DAYS = "mon-fri"   # Days to run: "mon-fri", "mon-sun", etc.
SCHEDULE_INTERVAL_MINUTES = 2  # Set > 0 to use interval mode instead of cron
