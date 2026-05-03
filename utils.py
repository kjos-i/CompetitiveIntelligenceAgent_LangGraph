"""Shared utility helpers for the CI agent.

Provides:
- setup_logger           — configure a dual file+console logger.
- print_agent_graph      — render the compiled agent graph as a PNG.
- load_watchlist         — parse and validate watchlist.json.
- compute_semantic_distance — cosine distance between two text strings.
"""

# Standard library
import json
import logging
from pathlib import Path

# Third-party
import numpy as np

# Local
from config import LOGGER_NAME
from pydantic_models import Company  # noqa: F401 — re-exported for callers


DEFAULT_LOGGER_NAME = LOGGER_NAME


# --- Visualization ---
def print_agent_graph(agent, filename="agent_graph.png", logger=None):
    """Render the compiled LangGraph agent as a PNG and save it to *filename*.

    Uses the Mermaid.js API embedded in LangGraph to draw nodes and edges.
    Only called when DRAW = True in config.py; safe to skip in production.

    Args:
        agent:    The compiled LangGraph CompiledStateGraph instance.
        filename: Destination path for the PNG file.
        logger:   Optional logger; falls back to the module-level logger.
    """
    log = logger or logging.getLogger(DEFAULT_LOGGER_NAME)

    try:
        # Retrieve the internal graph structure from the compiled agent
        agent_graph = agent.get_graph()

        # Generate binary PNG data using the Mermaid API
        png_bytes = agent_graph.draw_mermaid_png()

        # Save the byte stream to a physical file
        with open(filename, "wb") as f:
            f.write(png_bytes)

        log.info(f"Agent graph saved to {filename}.")

    except Exception as e:
        log.warning(f"Could not generate graph: {e}")


# --- Logging Infrastructure ---
def setup_logger(name=DEFAULT_LOGGER_NAME, log_file=None, console=False):
    """Return a named logger with a rotating file handler and optional console handler.

    Safe to call multiple times for the same *name*: handlers are only added
    once, so repeated calls return the same configured logger.

    Args:
        name:      Logger name; defaults to LOGGER_NAME from config.py.
        log_file:  Path to the log file.  Defaults to agent.log next to
                   this module if not provided.
        console:   When True, also echo INFO+ messages to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        log_path = Path(log_file) if log_file else Path(__file__).resolve().parent / "agent.log"

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

    return logger


def load_watchlist(filename="watchlist.json") -> list[Company]:
    """Load and validate the competitor watchlist from *filename*.

    Each entry in the JSON competitors array is validated against the
    Company Pydantic model.  Invalid entries are skipped with a warning.
    Returns an empty list if the file is missing or unreadable.
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    try:
        path = Path(__file__).resolve().parent / filename
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        companies = []
        for entry in data.get("competitors", []):
            try:
                companies.append(Company.model_validate(entry))
            except Exception as e:
                logger.warning(f"Skipping invalid watchlist entry {entry}: {e}")
        return companies
    except FileNotFoundError:
        logger.warning(f"Watchlist file not found: {filename}")
        return []
    except Exception as e:
        logger.error(f"Error loading watchlist from {filename}: {e}")
        return []


async def compute_semantic_distance(prev_text: str, new_text: str, embeddings) -> float | None:
    """Return the cosine distance between *prev_text* and *new_text*.

    A value of 0.0 means the texts are identical; 1.0 means they are
    completely dissimilar.  Returns None when either text is empty or
    when the embeddings call fails.

    Args:
        prev_text:  The earlier text (e.g. PREVIOUS STATUS report).
        new_text:   The newer text (e.g. the current research report).
        embeddings: A LangChain embeddings instance that supports
                    aembed_documents.
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    if not prev_text or not new_text:
        return None
    try:
        vecs = await embeddings.aembed_documents([prev_text, new_text])
        prev_vec = np.array(vecs[0])
        new_vec = np.array(vecs[1])
        cosine_sim = np.dot(prev_vec, new_vec) / (np.linalg.norm(prev_vec) * np.linalg.norm(new_vec))
        return round(float(1 - cosine_sim), 4)
    except Exception as e:
        logger.warning(f"Semantic distance calculation failed: {e}")
        return None
