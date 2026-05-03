"""Data models for the CI agent evaluation harness.

Two separate case types reflect the two evaluation layers:

  L1EvalCase — Layer 1 sub-agent isolation.  Calls brave_agent or
    tavily_agent directly (always live).  Tests structural properties of
    the tool response: result count, domain exclusion, domain inclusion.

  L2EvalCase — Layer 2 end-to-end agent run.  Supervisor + one sub-agent, forced
    by DIRECTIVE in the query.  Supports fixture mode (pre-recorded tool
    output) and live mode (real API calls).  Tests report format, delta
    scoring, citation presence, stop-rule compliance, and LLM-judged quality.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class L1EvalCase(BaseModel):
    """Layer 1: sub-agent isolation test case.  Always runs live."""

    id: str
    engine: Literal["brave", "tavily"]
    """Which sub-agent to call directly."""

    query: str
    """Raw search query sent directly to the sub-agent (no DIRECTIVE prefix needed)."""

    expected_min_results: int = Field(1, ge=1)
    """Minimum number of search results expected back from the tool."""

    required_keywords: list[str] = Field(default_factory=list)
    """Keywords that must appear somewhere in the sub-agent's final response text."""

    disallowed_keywords: list[str] = Field(default_factory=list)
    """Keywords that must NOT appear in the sub-agent's final response text."""

    category: str = ""
    notes: str = ""


class L2EvalCase(BaseModel):
    """Layer 2: end-to-end agent run test case (supervisor + one sub-agent)."""

    id: str
    engine: Literal["brave", "tavily"]
    """Which search engine the DIRECTIVE forces."""

    query: str
    """Full query string including 'DIRECTIVE: Use X only. TODAY: ... Research Y: ...'."""

    history_context: str
    """PREVIOUS STATUS string injected into the supervisor prompt via config['configurable']."""

    company: str
    """Canonical company name (used for context only; not passed to the agent separately)."""

    scenario_type: str
    """One of: no_change | minor_update | new_milestone | major_new_entity |
    old_information | no_prior_data | citation_required | stop_rule |
    directive_brave | directive_tavily."""

    expected_significance_range: list[int] = Field(
        default_factory=lambda: [1, 10],
        min_length=2,
        max_length=2,
        description="[min, max] inclusive. [1, 10] means any score is acceptable.",
    )

    expected_max_tool_calls: int = Field(
        default=1,
        ge=1,
        description=(
            "Maximum number of tool calls allowed before the case is gated as "
            "REVIEW (routing toggle group). Defaults to 1 to enforce the system "
            "prompt's Stop Rule."
        ),
    )

    expected_answer_points: list[str] = Field(
        default_factory=list,
        description=(
            "Key facts the report should cover. Used as the gold-standard "
            "context/expected_output for GEval LLM judge metrics."
        ),
    )

    required_keywords: list[str] = Field(default_factory=list)
    disallowed_keywords: list[str] = Field(default_factory=list)

    fixture: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Hand-authored agent output for fixture mode. Expected keys:\n"
            "  final_report    (str)        — Supervisor's synthesized report text.\n"
            "  tool_names      (list[str])  — Verbatim ordered list of tool calls\n"
            "                                 made by the supervisor; drives\n"
            "                                 directive_compliance and\n"
            "                                 tool_call_count in fixture replays.\n"
            "  tool_output         (str)        — Concatenated ToolMessage content\n"
            "                                     used as retrieval_context for judges.\n"
            "  latency_seconds     (float)      — Optional illustrative latency.\n"
            "Authored alongside ``history_context`` and ``expected_answer_points``\n"
            "to form a self-consistent synthetic scenario. Null means no fixture\n"
            "for this case; it will be skipped in fixture mode."
        ),
    )

    category: str = ""
    notes: str = ""
