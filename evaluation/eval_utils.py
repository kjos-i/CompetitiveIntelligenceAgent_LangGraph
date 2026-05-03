"""Utility helpers for the CI agent evaluation harness.

Covers: data loading, text extraction from LangChain messages, DeepEval
context builders, and generic math helpers.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from eval_pydantic_models import L1EvalCase, L2EvalCase


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Lowercase, strip accents, collapse punctuation for substring matching."""
    if not value:
        return ""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower()
    ascii_value = re.sub(r"[^a-z0-9.%\-\s]", " ", ascii_value)
    return re.sub(r"\s+", " ", ascii_value).strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_l1_cases(dataset_path: Path) -> list[L1EvalCase]:
    """Load and validate Layer 1 evaluation cases from a JSON file."""
    if not dataset_path.exists():
        raise SystemExit(f"Layer 1 dataset not found: {dataset_path}")
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        return [L1EvalCase.model_validate(item) for item in payload]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise SystemExit(f"Invalid Layer 1 dataset '{dataset_path}': {exc}") from exc


def load_l2_cases(dataset_path: Path) -> list[L2EvalCase]:
    """Load and validate Layer 2 evaluation cases from a JSON file."""
    if not dataset_path.exists():
        raise SystemExit(f"Layer 2 dataset not found: {dataset_path}")
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        return [L2EvalCase.model_validate(item) for item in payload]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise SystemExit(f"Invalid Layer 2 dataset '{dataset_path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Text extraction from LangChain messages
# ---------------------------------------------------------------------------

def extract_message_text(content: Any) -> str:
    """Extract plain text from a LangChain message content field.

    Handles both the simple string case and the multi-modal block list
    case (where content is a list of dicts each carrying a text
    or content key).
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def extract_all_tool_outputs(messages: list[Any]) -> list[str]:
    """Return the content of every ToolMessage in a message list.

    Non-string content is JSON-serialised so downstream regex / substring
    checks always operate on text.
    """
    outputs: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            content = message.content
            if not content:
                continue
            outputs.append(content if isinstance(content, str) else json.dumps(content))
    return outputs


def content_to_text(value: Any) -> str:
    """Flatten a LangGraph message content payload into a plain string.

    Used in the L2 live path to read the supervisor's final report off
    its last AIMessage. Distinct from :func:extract_message_text:
    this helper preserves whitespace, joins block lists with no
    separator (markdown-friendly), and returns "" for None
    rather than the literal string "None".

    A copy of agent._content_to_text lives here intentionally — the
    eval harness owns its own message-parsing semantics so a future
    refactor in agent.py cannot silently change how reports are
    interpreted by the judges.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
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


def extract_messages(output: Any) -> list[Any]:
    """Pull a messages list off a LangGraph node output, however it's shaped.

    Accepts either a state dict ({"messages": [...]}) or an object
    exposing a messages attribute. Returns [] when neither is
    present rather than raising, because LangGraph node outputs vary
    by node type and the eval should treat "no messages" as a clean
    empty result.

    A copy of agent._extract_messages lives here intentionally —
    see :func:content_to_text for the rationale.
    """
    if isinstance(output, dict):
        return output.get("messages", []) or []
    return getattr(output, "messages", []) or []


def join_tool_outputs(outputs: list[str]) -> str:
    """Join collected tool outputs into the single blob judges expect.

    The L1 and L2 live paths capture tool outputs differently — L1
    reads the finished messages list (extract_all_tool_outputs),
    L2 live appends to a list as on_tool_end events stream — but
    both feed this helper to produce the tool_output field.
    Centralising the join makes the parity between the two paths
    obvious and gives one place to change the separator if needed.
    """
    return "\n".join(outputs)

# ---------------------------------------------------------------------------
# DeepEval context builders (Layer 2)
# ---------------------------------------------------------------------------

def build_expected_output(case: L2EvalCase) -> str:
    """Build the expected_output string for DeepEval from the case's answer points.

    Used by judges that compare the agent's report against a target answer
    (relevancy, correctness). Falls back to a generic instruction when no
    explicit answer points are authored on the case.
    """
    if case.expected_answer_points:
        return " ".join(case.expected_answer_points)
    return (
        "The report should identify what is genuinely NEW compared to PREVIOUS STATUS, "
        "provide a significance score consistent with the delta, and cite real URLs."
    )


def build_gold_context(case: L2EvalCase) -> list[str]:
    """Build the gold-standard context list passed to DeepEval.

    The first entry is always the case's history_context (labelled
    PRIOR STATE) so the delta_quality judge can verify that items the
    report presents as new are genuinely absent from the prior ledger
    entries, and vice versa. Remaining entries are the case's
    expected_answer_points (or sensible defaults) so the same context
    still serves as a gold reference for any future judge that needs it.
    """
    history_entry = (
        f"PRIOR STATE (already known to the agent before this run):\n"
        f"{case.history_context}"
    )
    if case.expected_answer_points:
        return [history_entry, *case.expected_answer_points]
    return [
        history_entry,
        f"Scenario: {case.scenario_type}.",
        f"Expected significance range: {case.expected_significance_range[0]}–{case.expected_significance_range[1]}.",
        "All Key Findings must include a source URL.",
    ]


def build_retrieval_context(tool_output: str) -> list[str]:
    """Wrap the raw tool output as the retrieval_context list DeepEval expects.

    Used by judges that check faithfulness — claims in the report must be
    traceable to what the search tool actually returned. DeepEval requires
    a list even for a single payload.
    """
    if not tool_output:
        return ["No search results available."]
    return [tool_output]


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

_FMT_PRECISION_RE = re.compile(r"\.(\d+)([f%])")


def precision_for(fmt: str) -> int:
    """Translate a Python format spec into the matching rounding precision.

    Mapping:
      .<n>f → n (store the same number of decimals shown).
      .<n>% → n + 2 (the percent format multiplies by 100 at
        display time, so two extra stored decimals are needed for the
        n decimals shown after the conversion).
      "d"   → 0.
      Anything unrecognized → 2 as a safe default.

    Lets callers translate a MetricDef.fmt (or its
    summary_avg_fmt) into how many decimals to round to before
    storing, keeping stored precision in lockstep with display
    precision.
    """
    match = _FMT_PRECISION_RE.match(fmt)
    if match:
        digits = int(match.group(1))
        return digits + 2 if match.group(2) == "%" else digits
    if fmt == "d":
        return 0
    return 2


def safe_mean(values: list[float | int | None], precision: int = 2) -> float | None:
    """Return the mean of non-None values, or None when no values are present.

    Returning None for the empty case (rather than 0.0) keeps "not
    evaluated" distinct from a real zero in summary tables and
    dashboards. Pass precision_for(registry_fmt) to align stored
    precision with the metric's display format.
    """
    numeric = [float(value) for value in values if value is not None]
    return round(sum(numeric) / len(numeric), precision) if numeric else None
