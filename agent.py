"""Core runtime for the Moodgruppen competitive-intelligence agent.

This module configures the model, search tools, supervisor graph, and the
`run_agent` coroutine shared by the manual and automated workflows.
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure Unicode symbols print correctly on Windows (CP1252 → UTF-8).
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import BraveSearch
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph_supervisor import create_supervisor
from pydantic_models import SentimentResult

from config import (
    BRAVE_EXTRA_SNIPPETS,
    BRAVE_FRESHNESS,
    BRAVE_MAX_RESULTS,
    DRAW,
    EVENTS,
    EXCLUDED_DOMAINS,
    INCLUDED_DOMAINS,
    LOGGER_NAME,
    MESSAGES,
    MODEL_NAME,
    SIGNIFICANCE_THRESHOLD,
    TAVILY_MAX_RESULTS,
    TAVILY_SEARCH_DEPTH,
    TAVILY_TIME_RANGE,
    TAVILY_TOPIC,
    TEMPERATURE,
    UPDATES,
)
from memory_sqlite3 import init_db, save_to_ledger
from utils import print_agent_graph, setup_logger


# Load environment variables for local development.
load_dotenv()

# Configure shared logging and persistence.
logger = setup_logger(LOGGER_NAME, log_file=Path(__file__).with_name("agent.log"))

try:
    init_db()
    logger.info("Long-term memory cache initialized.")
except Exception as e:
    logger.warning(f"Could not initialize memory cache: {e}")

# Create the primary model and specialist agents.
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


# ---------------------------------------------------------------------------
# Sub-agent: broad web scout
# Uses Brave Search for fast, wide coverage of recent news and press activity.
# ---------------------------------------------------------------------------
_brave_prompt = """\
You are a web research scout. Search broadly and return any content related to the query.
Include content from sources such as official company websites, news outlets, press releases, financial press, and social media.  
Focus on recent developments and changes.
Do not include AI-generated summaries, content aggregators, or pages that are themselves summaries of other articles.
Do not include results that are historical and not relevant to the current state of the company.
"""

if EXCLUDED_DOMAINS:
    exclusion_list = ", ".join(EXCLUDED_DOMAINS)
    _brave_prompt += f" Never include results from: {exclusion_list}. Omit these sources entirely."

if INCLUDED_DOMAINS:
    inclusion_list = ", ".join(INCLUDED_DOMAINS)
    _brave_prompt += f" Prefer results from: {inclusion_list}. Prioritise these sources above others."

brave_agent = create_agent(
    llm,
    tools=[
        BraveSearch.from_api_key(
            api_key=os.getenv("BRAVE_SEARCH_API_KEY"),
            search_kwargs={
                "count": BRAVE_MAX_RESULTS,
                "freshness": BRAVE_FRESHNESS,
                "extra_snippets": BRAVE_EXTRA_SNIPPETS,
            },
        )
    ],
    name="brave_scout",
    system_prompt=_brave_prompt,
)

# ---------------------------------------------------------------------------
# Sub-agent: deep research analyst
# Uses Tavily for verified, source-level deep dives and fact-checking.
# ---------------------------------------------------------------------------
_tavily_prompt = """\
You are a research analyst. Given a query, conduct a throrough investigation on the topic.
Include content from sources such as official company websites, news outlets, press releases, financial press, and social media.
Focus on recent developments and changes.
Do not include AI-generated summaries, content aggregators, or pages that are themselves summaries of other articles.
Do not include results that are historical and not relevant to the current state of the company.
"""

tavily_agent = create_agent(
    llm,
    tools=[
        TavilySearch(
            max_results=TAVILY_MAX_RESULTS,
            search_depth=TAVILY_SEARCH_DEPTH,
            topic=TAVILY_TOPIC,
            time_range=TAVILY_TIME_RANGE,
            exclude_domains=EXCLUDED_DOMAINS,
            include_domains=INCLUDED_DOMAINS
        )
    ],
    name="tavily_analyst",
    system_prompt=_tavily_prompt,
)



# ---------------------------------------------------------------------------
# Supervisor graph
# Reads the routing logic from an external text file so the prompt can be
# edited without touching Python code.
# add_handoff_messages=False prevents the "Successfully transferred to ..."
# ToolMessage from being injected into the sub-agent's context, which would
# otherwise cause the LLM to comment on the handoff mechanism.
# ---------------------------------------------------------------------------
system_prompt_path = Path(__file__).resolve().with_name("system_prompt_agent.txt")
system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
logger.info(f"Loaded system prompt from: {system_prompt_path}")

supervisor = create_supervisor(
    [tavily_agent, brave_agent],
    model=llm,
    prompt=system_prompt,
    add_handoff_messages=False,
)
agent = supervisor.compile()
if DRAW:
    print_agent_graph(agent, filename=Path(__file__).with_name("agent_graph.png"), logger=logger)


def _content_to_text(value: Any) -> str:
    """Normalize LangChain/LangGraph content payloads into plain text safely."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_part = item.get("text") or item.get("content") or ""
                if text_part:
                    parts.append(str(text_part))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or value)
    return str(value)


def _extract_messages(output: Any) -> list[Any]:
    """Safely retrieve a messages list from different output shapes."""
    if isinstance(output, dict):
        return output.get("messages", []) or []
    return getattr(output, "messages", []) or []


async def run_agent(query: str, config: dict, company: str, engine: str, mode: str) -> bool:
    """Stream the agent graph for *query* and persist the result to the ledger.

    Iterates over LangGraph streaming events, printing live output to the
    console and collecting the final research report.  After the stream ends,
    a structured sentiment call is made and everything is saved to SQLite.

    Args:
        query:   Full prompt string sent to the supervisor (includes TODAY,
                 DIRECTIVE, PREVIOUS STATUS, etc.).
        config:  LangGraph runnable config; must contain a ``thread_id``.
        company: Canonical company name used as the ledger key.
        engine:  Either ``"brave_search"`` or ``"tavily"`` — stored for
                 filtering in the dashboard.
        mode:    Either ``"manual"`` or ``"auto"`` — stored for filtering.

    Returns:
        ``True`` if the significance score is at or above
        ``SIGNIFICANCE_THRESHOLD`` (i.e. a strategically notable change).
    """
    tool_timers: dict = {}   # run_id → wall-clock start time, used to compute tool latency
    significance_score = 0
    final_text: str | None = None  # Populated with the last supervisor output that contains SIGNIFICANCE_SCORE

    logger.info(f"→ Processing: {query}")

    events = agent.astream_events({"messages": [HumanMessage(content=query)]}, config=config, version="v2")

    async for event in events:
        if not isinstance(event, dict):
            logger.warning(f"Skipping unexpected event payload: {event!r}")
            continue

        event_name = event.get("event", "")
        run_id = event.get("run_id")
        event_data = event.get("data", {}) if isinstance(event.get("data", {}), dict) else {}
        event_label = event.get("name", "")

        if EVENTS:
            print(f"\n[Event] {event_name}", flush=True)
            logger.debug(f"Event: {event_name} | Run ID: {run_id} | Data keys: {list(event_data.keys())}")

        try:
            if UPDATES and event_name == "on_chain_stream":
                updates = event_data.get("updates", {})
                if isinstance(updates, dict):
                    for node, delta in updates.items():
                        logger.debug(f"Node {node} output: {delta}")
                        if node != "__metadata__":
                            print(f"◈ Node '{node}' active.")
                            sys.stdout.flush()

            if event_name == "on_chat_model_start":
                print("\n✧ Thinking...\n", end=" ", flush=True)

            if event_name == "on_chat_model_stream":
                chunk = event_data.get("chunk")
                content = _content_to_text(getattr(chunk, "content", None))
                if content:
                    print(content, end="", flush=True)

            if event_name == "on_tool_start":
                if run_id is not None:
                    tool_timers[run_id] = time.time()
                print(f"▶ Tool: {event_label}")
                logger.debug(f"Tool {event_label} input: {event_data.get('input')}")
                sys.stdout.flush()

            if event_name == "on_tool_end":
                start_time = tool_timers.pop(run_id, None) if run_id is not None else None
                latency = time.time() - start_time if start_time else 0
                print(f"⏱ {latency:.2f}s {event_label} finished.")
                logger.info(f"Tool {event_label} completed in {latency:.2f}s")
                sys.stdout.flush()

            if event_name == "on_tool_error":
                if run_id is not None:
                    tool_timers.pop(run_id, None)
                error_msg = _content_to_text(event_data.get("error", "Unknown error"))
                print(f"!! ERROR in {event_label}: {error_msg}")
                logger.error(f"Tool {event_label} failed: {error_msg}")
                sys.stdout.flush()

            if event_name == "on_chain_end" and event_label == "agent":
                final_output = event_data.get("output", {})
                messages = _extract_messages(final_output)

                if MESSAGES:
                    lines = ["--- FULL STATE ---"]
                    for i, msg in enumerate(messages):
                        role = type(msg).__name__
                        content = _content_to_text(getattr(msg, "content", msg))
                        lines.append(f"[{i}] {role}: {content}")
                    lines.append("--- END STATE ---")
                    logger.debug("\n".join(lines))

                if messages:
                    text = _content_to_text(getattr(messages[-1], "content", messages[-1]))
                    match = re.search(r"SIGNIFICANCE_SCORE\D+(\d+)", text)
                    # Only accept chain-end events that carry a completed research report
                    # (identified by the mandatory SIGNIFICANCE_SCORE marker).  This
                    # filters out intermediate events from sub-agents and the supervisor's
                    # internal routing steps, which do not contain the full report.
                    if match:
                        significance_score = int(match.group(1))
                        final_text = text

        except Exception as e:
            logger.warning(f"Skipping malformed event '{event_name or 'unknown'}': {e}")
            continue

    # --- Post-stream: save exactly once using the last valid research report ---
    if final_text is None:
        logger.warning("No agent output containing SIGNIFICANCE_SCORE was found — nothing saved.")
        return False

    sentiment_result = None
    try:
        structured_llm = llm.with_structured_output(SentimentResult)
        sentiment_result = await structured_llm.ainvoke([
            HumanMessage(content=f"""\
            From the report below, extract the following:
            - sentiment: In one word, describe the overall sentiment as positive, negative, neutral, or mixed.
            - sentiment_score: Rate the sentiment on an integer scale from 1-10 where 10 is most positive.

            Report:
            {final_text}""")
        ])

    except Exception as e:
        logger.warning(f"Structured sentiment extraction failed: {e}")

    sentiment = sentiment_result.sentiment if sentiment_result else "unknown"
    sentiment_score = sentiment_result.sentiment_score if sentiment_result else None
    is_major_change = significance_score >= SIGNIFICANCE_THRESHOLD

    try:
        save_to_ledger(
            company,
            query,
            final_text,
            significance_score,
            significance_flag=is_major_change,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            engine=engine,
            mode=mode,
        )
    except Exception as e:
        logger.warning(f"Could not save to ledger: {e}")

    print("\n" + "─" * 40)
    logger.info(f"Task complete. Score: {significance_score} | Significant: {is_major_change}")

    return is_major_change
