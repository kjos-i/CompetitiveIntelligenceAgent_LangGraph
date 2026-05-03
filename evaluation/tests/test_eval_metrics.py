"""Unit tests for the deterministic compute_* functions."""

from __future__ import annotations

from eval_metrics import (
    compute_citation_presence,
    compute_directive_compliance,
    compute_domain_exclusion_respected,
    compute_domain_inclusion_present,
    compute_keyword_checks,
    compute_report_length,
    compute_required_sections_present,
    compute_result_URL_count,
    compute_score_in_expected_range,
    compute_significance_score,
    compute_significance_score_valid,
    compute_tool_call_count,
    extract_urls_from_text,
)

# Sample report with all required sections, used as the "happy path" baseline.
GOOD_REPORT = """\
## Executive Summary
A short summary.

## Key Findings
- Acme launched a new product (https://example.com/news/1).
- Beta raised a Series C round (https://example.com/news/2).

## Competitive Implication
Some implications.

## Sources
- https://example.com/news/1
- https://example.com/news/2

## Significance Score Reasoning
Reasoning here.
SIGNIFICANCE_SCORE: 7
"""


# ---------------------------------------------------------------------------
# Format-compliance metrics
# ---------------------------------------------------------------------------

class TestRequiredSectionsPresent:
    def test_all_present_scores_one(self):
        assert compute_required_sections_present(GOOD_REPORT) == 1.0

    def test_missing_section_scores_partial(self):
        partial = GOOD_REPORT.replace("## Sources", "## Other")
        score = compute_required_sections_present(partial)
        assert 0.0 < score < 1.0

    def test_empty_report(self):
        assert compute_required_sections_present("") == 0.0


class TestCitationPresence:
    def test_all_bullets_have_url(self):
        assert compute_citation_presence(GOOD_REPORT) == 1.0

    def test_no_url_bullets(self):
        report = (
            "## Key Findings\n"
            "- Bullet without source\n"
            "- Another with no link\n"
            "## Competitive Implication\n"
        )
        assert compute_citation_presence(report) == 0.0

    def test_partial_citation(self):
        report = (
            "## Key Findings\n"
            "- With link (https://example.com/a)\n"
            "- Without link\n"
            "## Competitive Implication\n"
        )
        assert compute_citation_presence(report) == 0.5

    def test_empty_section_passes_vacuously(self):
        # No bullets means no citation rule to violate.
        report = "## Key Findings\n\n## Competitive Implication\n"
        assert compute_citation_presence(report) == 1.0

    def test_no_key_findings_section(self):
        assert compute_citation_presence("## Other\n- bullet\n") == 0.0

    def test_empty_report(self):
        assert compute_citation_presence("") == 0.0


class TestSignificanceScoreValid:
    def test_valid_score(self):
        assert compute_significance_score_valid("SIGNIFICANCE_SCORE: 5") == 1.0

    def test_score_at_lower_bound(self):
        assert compute_significance_score_valid("SIGNIFICANCE_SCORE: 1") == 1.0

    def test_score_at_upper_bound(self):
        assert compute_significance_score_valid("SIGNIFICANCE_SCORE: 10") == 1.0

    def test_score_above_range(self):
        assert compute_significance_score_valid("SIGNIFICANCE_SCORE: 11") == 0.0

    def test_score_below_range(self):
        assert compute_significance_score_valid("SIGNIFICANCE_SCORE: 0") == 0.0

    def test_missing_score(self):
        assert compute_significance_score_valid("nothing here") == 0.0

    def test_case_insensitive(self):
        assert compute_significance_score_valid("significance_score: 7") == 1.0


# ---------------------------------------------------------------------------
# Delta-scoring metrics
# ---------------------------------------------------------------------------

class TestSignificanceScore:
    def test_extracts_int(self):
        assert compute_significance_score("SIGNIFICANCE_SCORE: 7") == 7

    def test_returns_none_when_missing(self):
        assert compute_significance_score("no score") is None

    def test_returns_none_for_empty(self):
        assert compute_significance_score("") is None


class TestScoreInExpectedRange:
    def test_inside_range(self):
        assert compute_score_in_expected_range("SIGNIFICANCE_SCORE: 5", [3, 7]) == 1.0

    def test_at_boundaries(self):
        assert compute_score_in_expected_range("SIGNIFICANCE_SCORE: 3", [3, 7]) == 1.0
        assert compute_score_in_expected_range("SIGNIFICANCE_SCORE: 7", [3, 7]) == 1.0

    def test_outside_range(self):
        assert compute_score_in_expected_range("SIGNIFICANCE_SCORE: 2", [3, 7]) == 0.0
        assert compute_score_in_expected_range("SIGNIFICANCE_SCORE: 8", [3, 7]) == 0.0

    def test_missing_score_returns_zero(self):
        assert compute_score_in_expected_range("none", [1, 10]) == 0.0


# ---------------------------------------------------------------------------
# Routing metrics
# ---------------------------------------------------------------------------

class TestToolCallCount:
    def test_counts_all(self):
        assert compute_tool_call_count(["a", "b", "c"]) == 3

    def test_empty_list(self):
        assert compute_tool_call_count([]) == 0


class TestDirectiveCompliance:
    def test_match_scores_one(self):
        assert compute_directive_compliance("brave_scout", ["brave_scout"]) == 1.0

    def test_extra_calls_still_match(self):
        assert compute_directive_compliance("brave_scout", ["brave_scout", "other"]) == 1.0

    def test_mismatch(self):
        assert compute_directive_compliance("brave_scout", ["tavily_analyst"]) == 0.0

    def test_no_calls_with_expectation(self):
        assert compute_directive_compliance("brave_scout", []) == 0.0

    def test_no_expectation_passes_vacuously(self):
        assert compute_directive_compliance(None, ["any"]) == 1.0
        assert compute_directive_compliance("", ["any"]) == 1.0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestReportLength:
    def test_counts_chars(self):
        assert compute_report_length("hello") == 5

    def test_empty(self):
        assert compute_report_length("") == 0
        assert compute_report_length(None) == 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Keyword checks (composite)
# ---------------------------------------------------------------------------

class TestKeywordChecks:
    def test_all_required_hit(self):
        out = compute_keyword_checks("Acme launched a product", ["Acme", "product"], [])
        assert out["required_keyword_hit_rate"] == 1.0
        assert out["disallowed_keyword_hits"] == 0

    def test_partial_required(self):
        out = compute_keyword_checks("Acme launched", ["Acme", "product"], [])
        assert out["required_keyword_hit_rate"] == 0.5

    def test_no_required_keywords_passes_vacuously(self):
        out = compute_keyword_checks("anything", [], [])
        assert out["required_keyword_hit_rate"] == 1.0

    def test_disallowed_count(self):
        out = compute_keyword_checks(
            "I cannot answer that, I don't have access",
            [],
            ["I cannot", "I don't have"],
        )
        assert out["disallowed_keyword_hits"] == 2

    def test_normalization_handles_accents(self):
        # "café" in report should match "cafe" in required list.
        out = compute_keyword_checks("café opened", ["cafe"], [])
        assert out["required_keyword_hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# Layer 1 sub-agent isolation metrics
# ---------------------------------------------------------------------------

class TestResultURLCount:
    def test_counts_urls(self):
        assert compute_result_URL_count(["https://a.com", "https://b.com"]) == 2

    def test_empty(self):
        assert compute_result_URL_count([]) == 0


class TestDomainExclusionRespected:
    def test_no_excluded_domains(self):
        assert compute_domain_exclusion_respected(["https://x.com"], []) == 1.0

    def test_no_urls(self):
        assert compute_domain_exclusion_respected([], ["linkedin.com"]) == 1.0

    def test_excluded_domain_present_fails(self):
        assert compute_domain_exclusion_respected(
            ["https://linkedin.com/in/x"], ["linkedin.com"]
        ) == 0.0

    def test_clean_urls_pass(self):
        assert compute_domain_exclusion_respected(
            ["https://example.com/news"], ["linkedin.com"]
        ) == 1.0

    def test_case_insensitive_match(self):
        assert compute_domain_exclusion_respected(
            ["https://LinkedIn.com/x"], ["linkedin.com"]
        ) == 0.0


class TestDomainInclusionPresent:
    def test_returns_none_when_no_inclusion_list(self):
        # Empty inclusion list means "no policy" — distinct from "passed".
        assert compute_domain_inclusion_present(["https://x.com"], []) is None

    def test_returns_none_when_no_urls(self):
        # Empty URL list cannot satisfy a positive check.
        assert compute_domain_inclusion_present([], ["example.com"]) is None

    def test_match_scores_one(self):
        assert compute_domain_inclusion_present(
            ["https://example.com/a"], ["example.com"]
        ) == 1.0

    def test_no_match_scores_zero(self):
        assert compute_domain_inclusion_present(
            ["https://other.com/a"], ["example.com"]
        ) == 0.0


class TestExtractUrlsFromText:
    def test_extracts_https(self):
        assert extract_urls_from_text("see https://example.com here") == [
            "https://example.com"
        ]

    def test_extracts_http(self):
        assert extract_urls_from_text("http://example.com") == ["http://example.com"]

    def test_strips_trailing_punctuation(self):
        # The pattern stops at the first comma / paren / quote.
        assert extract_urls_from_text("(https://example.com)") == [
            "https://example.com"
        ]

    def test_multiple_urls(self):
        urls = extract_urls_from_text("a https://x.com b https://y.com c")
        assert urls == ["https://x.com", "https://y.com"]

    def test_empty_input(self):
        assert extract_urls_from_text("") == []
        assert extract_urls_from_text(None) == []  # type: ignore[arg-type]
