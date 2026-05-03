"""Deterministic metric functions for the CI agent evaluation harness.

All functions are pure: they take plain strings or lists and return floats,
ints, or dicts.  No LLM calls are made here.

Layer 2 metrics (report analysis)
----------------------------------
- compute_required_sections_present — fraction of mandatory sections found.
- compute_citation_presence         — fraction of Key Findings bullets with a URL.
- compute_significance_score_valid  — SIGNIFICANCE_SCORE present and in [1, 10].
- compute_significance_score        — extract the integer score (or None).
- compute_score_in_expected_range   — 1.0 if score falls within case's range.
- compute_tool_call_count           — number of tool calls made.
- compute_directive_compliance      — called tool matches DIRECTIVE in query.
- compute_keyword_checks            — required/disallowed keyword hits.

Layer 1 metrics (sub-agent tool response)
------------------------------------------
- compute_result_URL_count          — number of search results returned.
- compute_domain_exclusion_respected— no excluded domain in result URLs.
- compute_domain_inclusion_present  — at least one included domain present.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from eval_metric_registry import metric_fmts
from eval_utils import normalize_text, precision_for

# Snapshot of the registry's format table at import time (metric_fmts
# is @cache-decorated so a fresh import reflects the current registry).
_FMTS = metric_fmts()


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# SIGNIFICANCE_SCORE: <int> line emitted by the supervisor (case-insensitive).
_SIGNIFICANCE_PATTERN = re.compile(r"SIGNIFICANCE_SCORE\s*:\s*(\d+)", re.IGNORECASE)
# HTTP(S) URL up to the first whitespace, paren, quote, or comma.
_URL_PATTERN          = re.compile(r"https?://[^\s\)\"\'\,]+")
# Bullet line starting with -, •, or * after optional indent.
_BULLET_PATTERN       = re.compile(r"^\s*[-•*]\s+.+", re.MULTILINE)

# The supervisor's RESPONSE FORMAT is the source of truth for which sections
# the report must contain. We parse it from system_prompt_agent.txt at
# import time so the eval can never drift from the prompt: rename a section
# in one place and the format gate follows automatically. Each format-section
# line in the prompt looks like N. Section Name: [description]; the
# regex below captures everything between the leading ordinal and the colon.
_SYSTEM_PROMPT_PATH  = Path(__file__).resolve().parent.parent / "system_prompt_agent.txt"
_SECTION_LINE_REGEX  = re.compile(r"^\s*\d+\.\s+([^:[\n]+?):", re.MULTILINE)


def _load_required_sections(prompt_path: Path) -> list[str]:
    """Extract the report's mandatory section headers from the system prompt.

    Looks at the lines under the RESPONSE FORMAT heading where each
    section is listed as N. Section Name: [...]. Raises SystemExit
    if the file is missing or no sections were parsed — silently
    proceeding with an empty list would make every report trivially pass
    compute_required_sections_present (vacuously 1.0 with no required
    sections), masking a real configuration problem.
    """
    if not prompt_path.exists():
        raise SystemExit(
            f"System prompt not found at {prompt_path}; "
            "eval_metrics cannot derive the required-sections list."
        )
    text = prompt_path.read_text(encoding="utf-8")
    response_format_marker = "### RESPONSE FORMAT"
    marker_index = text.find(response_format_marker)
    section_block = text[marker_index:] if marker_index >= 0 else text
    sections = [match.group(1).strip() for match in _SECTION_LINE_REGEX.finditer(section_block)]
    if not sections:
        raise SystemExit(
            f"Failed to parse any required sections from {prompt_path}. "
            "Expected lines of the form 'N. Section Name: [...]' under "
            "'### RESPONSE FORMAT'."
        )
    return sections


_REQUIRED_SECTIONS = _load_required_sections(_SYSTEM_PROMPT_PATH)


# ---------------------------------------------------------------------------
# Layer 2 — format compliance metrics
# ---------------------------------------------------------------------------

def compute_required_sections_present(report: str) -> float:
    """Fraction of the 5 mandatory sections found in the report (0.0 – 1.0).

    Returns 1.0 only when all five section headers are present.
    """
    if not report:
        return 0.0
    report_lower = report.lower()
    found = sum(1 for section in _REQUIRED_SECTIONS if section.lower() in report_lower)
    return round(found / len(_REQUIRED_SECTIONS), precision_for(_FMTS["required_sections_present"]))


def compute_citation_presence(report: str) -> float:
    """Fraction of bullet points in the Key Findings section that contain a URL.

    A bullet without a URL fails the mandatory citation rule from the system prompt.
    Returns 1.0 when no bullets are present (nothing to check).
    """
    if not report:
        return 0.0
    # Anchor the end-of-section lookahead to the start of a line so that
    # words like "competitive implications" appearing inside a bullet do not
    # truncate the Key Findings section mid-content.
    key_findings_match = re.search(
        r"Key Findings(.*?)(?=^\s*(?:Competitive Implication|Sources|Significance Score Reasoning)\b|\Z)",
        report,
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    if not key_findings_match:
        return 0.0
    key_findings_text = key_findings_match.group(1)
    bullets = _BULLET_PATTERN.findall(key_findings_text)
    if not bullets:
        return 1.0  # No bullets — section exists but is empty; format is still valid.
    bullets_with_url = sum(1 for bullet in bullets if _URL_PATTERN.search(bullet))
    return round(bullets_with_url / len(bullets), precision_for(_FMTS["citation_presence"]))


def compute_significance_score_valid(report: str) -> float:
    """1.0 if SIGNIFICANCE_SCORE: X is present and X ∈ [1, 10], else 0.0."""
    if not report:
        return 0.0
    match = _SIGNIFICANCE_PATTERN.search(report)
    if not match:
        return 0.0
    score = int(match.group(1))
    return 1.0 if 1 <= score <= 10 else 0.0


# ---------------------------------------------------------------------------
# Layer 2 — delta scoring metrics
# ---------------------------------------------------------------------------

def compute_significance_score(report: str) -> int | None:
    """Extract the integer significance score from the report text.

    Returns None if the SIGNIFICANCE_SCORE line is absent or malformed.
    """
    if not report:
        return None
    match = _SIGNIFICANCE_PATTERN.search(report)
    if not match:
        return None
    return int(match.group(1))


def compute_score_in_expected_range(report: str, expected_range: list[int]) -> float:
    """1.0 if the extracted score falls within [min, max] inclusive, else 0.0.

    Returns 0.0 when no score can be extracted.
    """
    score = compute_significance_score(report)
    if score is None:
        return 0.0
    low, high = expected_range[0], expected_range[1]
    return 1.0 if low <= score <= high else 0.0


# ---------------------------------------------------------------------------
# Layer 2 — routing / behavior metrics
# ---------------------------------------------------------------------------

def compute_tool_call_count(tool_names_called: list[str]) -> int:
    """Return the total number of tool calls made during the agent run."""
    return len(tool_names_called)


def compute_directive_compliance(expected_tool_name: str | None, tool_names_called: list[str]) -> float:
    """1.0 if the expected sub-agent was actually called, else 0.0.

    expected_tool_name is derived from the case's engine field
    (validated by Pydantic) rather than substring-matched out of the query,
    so a case whose query happens to mention the tool name for unrelated
    reasons does not get false-credited as compliant.

    Returns 1.0 when expected_tool_name is None or empty (nothing to enforce).
    Returns 0.0 when no tool was called at all and an expectation was set.
    """
    if not expected_tool_name:
        return 1.0
    if not tool_names_called:
        return 0.0
    return 1.0 if expected_tool_name in tool_names_called else 0.0


def compute_report_length(report: str) -> int:
    """Character count of the final report.

    Cheap diagnostic that catches truncated or empty outputs before any judge
    has to reason about them. Always-on regardless of the format toggle.
    """
    if not report:
        return 0
    return len(report)


# ---------------------------------------------------------------------------
# Keyword checks (shared, always-on)
# ---------------------------------------------------------------------------

def compute_keyword_checks(
    report: str,
    required_keywords: list[str],
    disallowed_keywords: list[str],
) -> dict[str, Any]:
    """Return required-keyword hit rate and disallowed-keyword hit count.

    Both checks use normalized substring matching so accents and punctuation
    do not cause false misses.
    """
    normalized_report = normalize_text(report)

    if required_keywords:
        hits = sum(1 for keyword in required_keywords if normalize_text(keyword) in normalized_report)
        required_keyword_hit_rate = round(hits / len(required_keywords), precision_for(_FMTS["required_keyword_hit_rate"]))
    else:
        required_keyword_hit_rate = 1.0

    disallowed_keyword_hits = sum(
        1 for keyword in disallowed_keywords if normalize_text(keyword) in normalized_report
    )

    return {
        "required_keyword_hit_rate": required_keyword_hit_rate,
        "disallowed_keyword_hits": disallowed_keyword_hits,
    }


# ---------------------------------------------------------------------------
# Layer 1 — sub-agent isolation metrics
# ---------------------------------------------------------------------------

def compute_result_URL_count(result_urls: list[str]) -> int:
    """Return the number of distinct result URLs extracted from the tool response."""
    return len(result_urls)


def compute_domain_exclusion_respected(result_urls: list[str], excluded_domains: list[str]) -> float:
    """1.0 if none of the result URLs contain an excluded domain, else 0.0.

    Returns 1.0 when no excluded domains are configured.
    """
    if not excluded_domains or not result_urls:
        return 1.0
    for url in result_urls:
        if any(domain.lower() in url.lower() for domain in excluded_domains):
            return 0.0
    return 1.0


def compute_domain_inclusion_present(result_urls: list[str], included_domains: list[str]) -> float | None:
    """1.0 if at least one result URL matches an included domain, else 0.0.

    Returns None when there's nothing to evaluate — either the inclusion
    list is empty (no policy configured) or there are no result URLs to
    check (zero results came back). Returning None instead of 1.0
    keeps "not evaluated" distinct from "vacuously passed" in the dashboard
    and stops a misleading positive score from being averaged into trends.
    Inclusion is a positive check, so empty results genuinely cannot
    satisfy it — unlike exclusion (a negative check), where empty results
    are vacuously safe.
    """
    if not included_domains or not result_urls:
        return None
    for url in result_urls:
        if any(domain.lower() in url.lower() for domain in included_domains):
            return 1.0
    return 0.0


def extract_urls_from_text(text: str) -> list[str]:
    """Extract all HTTP/HTTPS URLs found in a text blob.

    Used to parse raw tool output from BraveSearch or TavilySearch ToolMessages
    for Layer 1 domain-filter checks.
    """
    return _URL_PATTERN.findall(text or "")
