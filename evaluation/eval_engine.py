"""Evaluation engine for the CI agent harness.

Two layers, each with its own evaluation method:

  Layer 1 — evaluate_l1_case()
    Calls brave_agent or tavily_agent directly (always live).
    Tests structural properties of the raw search tool response:
    result count, domain exclusion, domain inclusion.
    No LLM judge metrics. No fixture support.

  Layer 2 — evaluate_l2_case()
    End-to-end agent run (supervisor + sub-agent) forced by a DIRECTIVE prefix.
    Supports two execution modes, selected per-engine via the
    execution_mode constructor arg (the runner builds one engine per
    RUN_VARIANTS entry):
      - "fixture": Loads pre-recorded output from case.fixture (the
        fixtures are hand-authored alongside the case's history_context
        and expected_answer_points, forming a self-consistent synthetic
        scenario). Skips API calls, runs the full metric panel (including
        delta judgments) against the stored report.
      - "live": Calls the real agent with real search APIs. Strips
        the synthetic history_context (passes empty so the agent runs
        as a cold-start research task) and skips the metrics that depend
        on synthetic ground truth: delta_quality,
        significance_justification (both encode old-vs-new delta
        judgments that collapse without prior state), and
        score_in_expected_range. expected_output and the gold
        context are also dropped, since the live report won't match
        the synthetic narrative they encode.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eval_metric_registry import fixture_only_metric_keys, judge_metric_keys, metric_fmts
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
from eval_pydantic_models import L1EvalCase, L2EvalCase
from eval_utils import (
    build_expected_output,
    build_gold_context,
    build_retrieval_context,
    content_to_text,
    extract_all_tool_outputs,
    extract_message_text,
    extract_messages,
    join_tool_outputs,
    precision_for,
    safe_mean,
)

# Snapshot of the registry's format table at import time. metric_fmts
# is @cache-decorated so a fresh import always reflects the current
# registry. Mutating METRICS at runtime requires reloading this module.
_FMTS = metric_fmts()

try:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
except ImportError as exc:
    raise SystemExit(
        "This script requires DeepEval. Install it with: pip install deepeval"
    ) from exc

from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Agent imports — add parent directory to path so agent.py resolves.
# Both agent.py and memory_sqlite3.py resolve their own paths via
# Path(__file__).resolve(), so importing here must not mutate the
# caller's working directory.
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

from agent import (  # noqa: E402
    agent as ci_agent,
    brave_agent,
    tavily_agent,
)
from config import EXCLUDED_DOMAINS, INCLUDED_DOMAINS  # noqa: E402


# Name of the compiled top-level supervisor graph emitted in
# astream_events on_chain_end events. If agent.py renames the
# supervisor, the live pipeline would silently capture an empty
# final_report; _run_agent_live records every chain-end name seen
# and emits a diagnostic when this expected name never appears. Verifying
# the compiled graph's .name at import time isn't reliable — that
# attribute defaults to "LangGraph" regardless of what's streamed —
# so the runtime check is authoritative.
_SUPERVISOR_EVENT_NAME = "agent"

# Map L2EvalCase.engine values to the supervisor's handoff tool names.
# LangGraph's supervisor exposes each sub-agent as a tool prefixed with
# "transfer_to_" — that's the name that lands in astream_events as
# on_tool_start, not the bare agent name. Reading directly off the agent
# objects keeps eval and agent in lockstep: renaming a sub-agent in
# agent.py propagates here without silently scoring 0.0 on
# directive_compliance.
_ENGINE_TO_TOOL_NAME: dict[str, str] = {
    "brave":  f"transfer_to_{brave_agent.name}",
    "tavily": f"transfer_to_{tavily_agent.name}",
}


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """Runs both layers of the CI evaluation harness."""

    # Ordered keys returned by gate_thresholds() and persisted to
    # eval_runs.gate_thresholds. Exposed as a class attribute so the
    # dashboard can iterate the canonical list (preserving display order)
    # without duplicating the names.
    GATE_THRESHOLD_KEYS: tuple[str, ...] = (
        "judge_threshold",
        "required_keyword_threshold",
        "min_report_length",
    )

    def __init__(
        self,
        *,
        judge_model: str,
        threshold: float,
        required_keyword_threshold: float = 0.5,
        min_report_length: int = 200,
        enabled_groups: set[str] | None = None,
        execution_mode: str = "fixture",
        judge_concurrency: int = 8,
    ) -> None:
        """Construct the engine.

        Validates execution_mode and judge_concurrency up front so a
        misspelled mode never silently runs the wrong pipeline (which would
        burn API budget and pollute downstream filenames / ledger columns),
        and so a non-positive judge concurrency is caught before the first
        judge call hangs forever waiting on a zero-permit semaphore.
        """

        if execution_mode not in ("fixture", "live"):
            raise ValueError(
                f"execution_mode must be 'fixture' or 'live', got {execution_mode!r}."
            )
        if judge_concurrency < 1:
            raise ValueError(
                f"judge_concurrency must be >= 1, got {judge_concurrency}."
            )
        self.judge_model = judge_model
        self.threshold = threshold
        self.required_keyword_threshold = required_keyword_threshold
        self.min_report_length = min_report_length
        self.enabled_groups: set[str] = set(enabled_groups) if enabled_groups is not None else {"judge", "format", "routing"}
        self.execution_mode = execution_mode
        self._judge_semaphore = asyncio.Semaphore(judge_concurrency)

    def gate_thresholds(self) -> dict[str, float]:
        """Return the active threshold values persisted with each run.

        Stored in the eval_runs.gate_thresholds JSON column so the
        dashboard can warn when thresholds drift between runs (which would
        invalidate trend comparisons). Key order matches
        GATE_THRESHOLD_KEYS so the dashboard's display order stays
        canonical without duplicating the names.
        """
        values = {
            "judge_threshold": self.threshold,
            "required_keyword_threshold": self.required_keyword_threshold,
            "min_report_length": self.min_report_length,
        }
        return {key: values[key] for key in self.GATE_THRESHOLD_KEYS}

    # ------------------------------------------------------------------
    # Layer 1 — sub-agent isolation
    # ------------------------------------------------------------------

    async def run_l1_subagent(
        self,
        case: L1EvalCase,
    ) -> tuple[str, str, float, int | None, int | None, int, bool, str | None]:
        """Invoke brave_agent or tavily_agent directly for a layer 1 case.

        Returns (response_text, raw_tool_output, latency_seconds,
        input_tokens, output_tokens, tool_call_count,
        recursion_limit_hit, error_or_None).

        Token counts are summed across every AIMessage returned by
        the sub-agent (LangChain's usage_metadata, populated by
        ChatOpenAI and most modern providers). Returns None/None
        when the provider doesn't expose usage and on exception, so the
        dashboard can render "not evaluated" instead of a misleading
        zero. tool_call_count is the number of ToolMessage
        instances in the agent's response (how many times the ReAct
        loop invoked the search tool); 0 on exception.
        recursion_limit_hit is True when LangGraph's
        GraphRecursionError fired — kept distinct from a generic
        crash so the dashboard can chart it separately.

        Each call uses a unique thread_id so the agent's memory
        starts clean per case and cannot cross-contaminate runs.
        """
        if case.engine == "brave":
            target_agent = brave_agent
        elif case.engine == "tavily":
            target_agent = tavily_agent
        else:
            raise ValueError(
                f"Unknown engine {case.engine!r} on L1 case {case.id!r}; "
                "expected 'brave' or 'tavily'."
            )
        started = time.perf_counter()
        try:
            result = await target_agent.ainvoke(
                {"messages": [HumanMessage(content=case.query)]},
                config={"configurable": {"thread_id": f"l1-{case.id}-{uuid4().hex}"}},
            )
            messages = result.get("messages", [])
            response_text = extract_message_text(messages[-1].content if messages else "")
            # Collect every ToolMessage so retries / multi-call ReAct
            # loops show up in result_count and domain checks rather
            # than getting truncated to the first call. The shared
            # join_tool_outputs helper produces the final blob —
            # L2 live builds the same shape from streaming events
            # (see _run_agent_live); the collection mechanism
            # differs because L1 uses ainvoke and L2 uses
            # astream_events, but both paths feed the same join.
            tool_output = join_tool_outputs(extract_all_tool_outputs(messages))
            # Count ToolMessage instances directly rather than reusing
            # extract_all_tool_outputs's length: That helper skips
            # empty-content messages, which would undercount calls when a
            # tool returns no payload (still a real call, still counts).
            tool_call_count = sum(1 for m in messages if isinstance(m, ToolMessage))
            input_tokens, output_tokens = _sum_message_tokens(messages)
            latency = round(time.perf_counter() - started, precision_for(_FMTS["latency_seconds"]))
            return response_text, tool_output, latency, input_tokens, output_tokens, tool_call_count, False, None
        except Exception as exc:
            latency = round(time.perf_counter() - started, precision_for(_FMTS["latency_seconds"]))
            recursion_limit_hit = type(exc).__name__ == "GraphRecursionError"
            return "", "", latency, None, None, 0, recursion_limit_hit, f"Sub-agent call failed: {type(exc).__name__}: {exc}"


    async def evaluate_l1_case(self, case: L1EvalCase) -> dict[str, Any]:
        """Run a single layer 1 eval case and return the result dict.

        Mirrors the layer 2 shape with errors and failure_reasons
        keys so the SQLite ledger and CSV export can treat both layers
        uniformly.
        """
        errors: list[str] = []

        (
            response_text, tool_output, latency,
            agent_input_tokens, agent_output_tokens,
            tool_call_count, recursion_limit_hit, error,
        ) = await self.run_l1_subagent(case)
        if error:
            errors.append(error)
        agent_total_tokens = (
            agent_input_tokens + agent_output_tokens
            if agent_input_tokens is not None and agent_output_tokens is not None
            else None
        )

        # Extract URLs from tool output (ToolMessage JSON) for domain checks.
        result_urls = extract_urls_from_text(tool_output)

        result_count = compute_result_URL_count(result_urls)
        domain_exclusion_respected = compute_domain_exclusion_respected(result_urls, EXCLUDED_DOMAINS)
        domain_inclusion_present = compute_domain_inclusion_present(result_urls, INCLUDED_DOMAINS)
        keyword_checks = compute_keyword_checks(response_text, case.required_keywords, case.disallowed_keywords)
        report_length = compute_report_length(response_text)

        failure_reasons: list[str] = []
        if errors:
            failure_reasons.append("run produced errors")
        if recursion_limit_hit:
            failure_reasons.append("agent hit recursion limit")
        if report_length < self.min_report_length:
            failure_reasons.append(
                f"report_length {report_length} < min_report_length {self.min_report_length}"
            )
        if result_count < case.expected_min_results:
            failure_reasons.append(
                f"result_count {result_count} < expected_min_results {case.expected_min_results}"
            )
        if domain_exclusion_respected < 1.0:
            failure_reasons.append("excluded domain present in results")
        if keyword_checks.get("disallowed_keyword_hits", 0) > 0:
            failure_reasons.append(f"disallowed_keyword_hits = {keyword_checks['disallowed_keyword_hits']}")
        hit_rate = keyword_checks.get("required_keyword_hit_rate", 1.0)
        if hit_rate < self.required_keyword_threshold:
            failure_reasons.append(
                f"required_keyword_hit_rate {hit_rate:.2f} < threshold {self.required_keyword_threshold}"
            )
        status = "PASS" if not failure_reasons else "REVIEW"

        return {
            "id":                         case.id,
            "layer":                      1,
            "category":                   case.category,
            "notes":                      case.notes,
            "question":                   case.query,
            "expected_output":            f"≥{case.expected_min_results} results, domain rules respected.",
            "answer":                     response_text,
            "latency_seconds":            latency,
            "agent_input_tokens":         agent_input_tokens,
            "agent_output_tokens":        agent_output_tokens,
            "agent_total_tokens":         agent_total_tokens,
            "judge_input_tokens":         None,
            "judge_output_tokens":        None,
            "judge_total_tokens":         None,
            "result_count":               result_count,
            "domain_exclusion_respected": domain_exclusion_respected,
            "domain_inclusion_present":   domain_inclusion_present,
            "tool_call_count":            tool_call_count,
            "keyword_checks":             keyword_checks,
            "judge_metrics":              {},
            "avg_judge_score":            None,
            "report_length":              report_length,
            "recursion_limit_hit":        1 if recursion_limit_hit else 0,
            "status":                     status,
            "failure_reasons":            failure_reasons,
            "errors":                     errors,
        }

    async def evaluate_l1_cases(
        self,
        cases: list[L1EvalCase],
        concurrency: int = 2,
    ) -> list[dict[str, Any]]:
        """Run all Layer 1 cases concurrently, bounded by the concurrency limit."""
        semaphore = asyncio.Semaphore(max(concurrency, 1))

        async def bounded(case: L1EvalCase) -> dict[str, Any]:
            async with semaphore:
                return await self.evaluate_l1_case(case)

        return await asyncio.gather(*(bounded(case) for case in cases))

    # ------------------------------------------------------------------
    # Layer 2 judge panel
    # ------------------------------------------------------------------

    def build_judge_metrics(self, judge_llm: _TokenTrackingJudgeLLM) -> dict[str, Any]:
        """Build the DeepEval LLM judging panel for layer 2.

        Returns an empty dict when the "judge" toggle group is disabled.
        Drops every metric flagged fixture_only=True in the registry
        (currently delta_quality and significance_justification)
        when execution_mode == "live". Those metrics read or rely on
        synthetic ground truth from the case's history_context and
        produce nonsense when judged against a live report grounded in
        real search results.

        Every metric is bound to the same judge_llm instance so that
        token usage from the entire panel accumulates onto a single
        wrapper that the caller can read after the panel finishes.
        """

        if "judge" not in self.enabled_groups:
            return {}

        metrics = {
            "answer_relevancy": AnswerRelevancyMetric(
                threshold=self.threshold,
                model=judge_llm,
            ),
            "faithfulness": FaithfulnessMetric(
                threshold=self.threshold,
                model=judge_llm,
            ),
            "delta_quality": GEval(
                name="Delta Quality",
                criteria=(
                    "The CONTEXT contains a 'PRIOR STATE' entry describing what was already "
                    "known to the agent before this run. Determine whether the ACTUAL_OUTPUT "
                    "correctly attributes findings as new vs. already-known relative to that "
                    "PRIOR STATE. "
                    "Penalise the report if any item already present in PRIOR STATE is "
                    "presented as a new finding or used to inflate the SIGNIFICANCE_SCORE. "
                    "Penalise the report if a genuinely new finding (absent from PRIOR STATE "
                    "but present in the report) is dismissed as already known. "
                    "The SIGNIFICANCE_SCORE must reflect only the truly new delta, not the "
                    "absolute importance of pre-existing information."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                    LLMTestCaseParams.CONTEXT,
                ],
                threshold=self.threshold,
                model=judge_llm,
            ),
            "significance_justification": GEval(
                name="Significance Justification",
                criteria=(
                    "Determine whether the SIGNIFICANCE_SCORE assigned is logically consistent with "
                    "the report's own Significance Score Reasoning section and Key Findings. "
                    "A score of 7+ requires a brand-new strategic entity not previously known. "
                    "A score of 1-2 requires that nothing meaningfully new was found. "
                    "The reasoning must explicitly justify the chosen score."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                threshold=self.threshold,
                model=judge_llm,
            ),
            "no_speculation": GEval(
                name="No Speculation",
                criteria=(
                    "Determine whether the report avoids speculating or stating facts that cannot be "
                    "sourced from the search tool results provided in the retrieval context. "
                    "Every factual claim in Key Findings must be traceable to the search results. "
                    "If a URL is not available for a fact, the report should omit it rather than "
                    "fabricate a placeholder URL or invent a source."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                ],
                threshold=self.threshold,
                model=judge_llm,
            ),
        }

        # Compare against the registry BEFORE dropping fixture-only metrics
        # in live mode — the registry describes the full panel; the live
        # drop is a runtime concern, not a registry mismatch.
        assert set(metrics.keys()) == set(judge_metric_keys()), (
            f"Registry/engine metric key mismatch: {set(metrics.keys()) ^ set(judge_metric_keys())}"
        )
        if self.execution_mode == "live":
            for key in fixture_only_metric_keys():
                metrics.pop(key, None)
        return metrics


    async def _run_judge_metric(
        self,
        metric_name: str,
        metric: Any,
        test_case: LLMTestCase,
    ) -> tuple[str, dict[str, Any]]:
        """Run a single DeepEval metric and return (name, result_dict)."""
        try:
            async with self._judge_semaphore:
                await metric.a_measure(test_case)
            score = getattr(metric, "score", None)
            passed = metric.is_successful() if hasattr(metric, "is_successful") else (
                bool(score and score >= getattr(metric, "threshold", 0.5))
            )
            return metric_name, {
                "score":     round(float(score), precision_for(_FMTS.get(metric_name, ".2f"))) if score is not None else None,
                "reason":    getattr(metric, "reason", ""),
                "passed":    bool(passed),
                "threshold": getattr(metric, "threshold", None),
            }
        except Exception as exc:
            # Transient API/auth/network failures must not silently masquerade
            # as the agent producing a bad answer. The structured failure dict
            # preserves the exception type and message in reason, which is
            # written verbatim to the JSON report under
            # results[i].judge_metrics[name].
            return metric_name, {
                "score":     None,
                "reason":    f"Metric failed: {type(exc).__name__}: {exc}",
                "passed":    False,
                "threshold": getattr(metric, "threshold", None),
            }

    async def _run_judge_metrics(
        self,
        test_case: LLMTestCase,
    ) -> tuple[dict[str, Any], int | None, int | None]:
        """Run the full judge metric panel concurrently and report aggregate token usage.

        Returns (metric_results, judge_input_tokens, judge_output_tokens).
        Token counts are summed across every judge LLM call made for this
        case via the shared _TokenTrackingJudgeLLM instance. Returns
        (None, None) for token counts when the judge group is disabled
        (no panel was built, no calls were made — distinct from "called
        but no usage metadata was returned" which would be 0/0).
        """
        if "judge" not in self.enabled_groups:
            return {}, None, None

        judge_llm = _TokenTrackingJudgeLLM(self.judge_model)
        metrics   = self.build_judge_metrics(judge_llm)
        if not metrics:
            return {}, None, None
        pairs = await asyncio.gather(
            *(self._run_judge_metric(name, metric, test_case) for name, metric in metrics.items())
        )
        return dict(pairs), judge_llm.input_tokens, judge_llm.output_tokens

    # ------------------------------------------------------------------
    # Layer 2 — end-to-end agent run
    # ------------------------------------------------------------------

    async def _run_agent_live(
        self,
        case: L2EvalCase,
    ) -> tuple[str, list[str], str, float, int | None, int | None, bool, list[str]]:
        """Run the full agent live and collect results.

        Returns:
            (final_report, tool_names_called, tool_output, latency_seconds,
             agent_input_tokens, agent_output_tokens, recursion_limit_hit,
             errors)

        Token counts are summed from every on_chat_model_end event's
        usage_metadata (LangChain's standard interface, populated by
        ChatOpenAI and most modern providers) — covers both the supervisor
        LLM and any sub-agent LLM calls in the same pipeline. Returns
        None/None when no event carried usage metadata (e.g. a
        non-OpenAI provider) and on exception, so the dashboard can render
        "not evaluated" instead of a misleading zero.

        errors is a list of strings — empty on a clean run. Includes the
        exception message on a crash, plus a supervisor-name diagnostic when
        the expected _SUPERVISOR_EVENT_NAME was never seen in the
        on_chain_end stream (the symptom that previously produced a silently
        empty final_report). All entries land in the per-case errors
        list and flow into the JSON report.
        """
        config = {
            "configurable": {
                "thread_id": f"eval-{case.id}-{uuid4().hex}",
                # Strip the case's synthetic history_context in live mode:
                # the agent would otherwise be told "you previously knew
                # about X" while real search results show unrelated current
                # findings, forcing it to reconcile a fake prior state
                # against real data and producing a confused report.
                # Cold-start lets the functional metrics judge a coherent
                # report instead.
                "history_context": "",
            }
        }
        started = time.perf_counter()
        tool_names: list[str] = []
        tool_outputs: list[str] = []
        chain_end_names: set[str]  = set()
        agent_input_tokens = 0
        agent_output_tokens = 0
        final_report = ""
        recursion_limit_hit = False

        try:
            events = ci_agent.astream_events(
                {"messages": [HumanMessage(content=case.query)]},
                config=config,
                version="v2",
            )
            async for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = event.get("event", "")
                event_name = event.get("name", "")
                event_data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

                if event_type == "on_tool_start":
                    tool_names.append(event_name)

                if event_type == "on_tool_end":
                    # Mirror of L1's extract_all_tool_outputs path,
                    # but reading from streaming events because the L2
                    # supervisor exposes its run via astream_events
                    # rather than a finished messages list.
                    raw = event_data.get("output", "")
                    if raw:
                        tool_outputs.append(str(raw))

                if event_type == "on_chat_model_end":
                    output = event_data.get("output")
                    usage = getattr(output, "usage_metadata", None)
                    if usage:
                        agent_input_tokens  += int(usage.get("input_tokens", 0) or 0)
                        agent_output_tokens += int(usage.get("output_tokens", 0) or 0)

                if event_type == "on_chain_end":
                    chain_end_names.add(event_name)
                    if event_name == _SUPERVISOR_EVENT_NAME:
                        messages = extract_messages(event_data.get("output", {}))
                        if messages:
                            final_report = content_to_text(getattr(messages[-1], "content", messages[-1]))

            latency = round(time.perf_counter() - started, precision_for(_FMTS["latency_seconds"]))
            # Same final shape as L1's run_l1_subagent — see
            # join_tool_outputs for why the two paths share this
            # join despite different upstream collection mechanisms.
            tool_output = join_tool_outputs(tool_outputs)

            errors: list[str] = []
            if _SUPERVISOR_EVENT_NAME not in chain_end_names:
                # The hardcoded event name didn't match anything LangGraph
                # actually emitted — final_report will be empty unless one of
                # the seen names is the right one. List them so the user can
                # update _SUPERVISOR_EVENT_NAME without guessing.
                seen = ", ".join(sorted(chain_end_names)) or "<none>"
                errors.append(
                    f"Expected on_chain_end event name {_SUPERVISOR_EVENT_NAME!r} "
                    f"not seen during run; final_report may be empty. "
                    f"Observed chain-end names: {seen}. Update "
                    f"_SUPERVISOR_EVENT_NAME at the top of eval_engine.py."
                )

            return final_report, tool_names, tool_output, latency, agent_input_tokens, agent_output_tokens, recursion_limit_hit, errors

        except Exception as exc:
            latency = round(time.perf_counter() - started, precision_for(_FMTS["latency_seconds"]))
            # LangGraph's GraphRecursionError surfaces as a distinct failure
            # mode (the agent looped past recursion_limit) — flag it so the
            # dashboard can chart it separately from generic errors.
            recursion_limit_hit = type(exc).__name__ == "GraphRecursionError"
            return "", [], "", latency, 0, 0, recursion_limit_hit, [
                f"Agent run failed: {type(exc).__name__}: {exc}"
            ]


    def _compute_status(
        self,
        judge_metrics: dict[str, Any],
        required_sections_present: float | None,
        citation_presence: float | None,
        significance_score_valid: float | None,
        score_in_expected_range: float | None,
        directive_compliance: float | None,
        tool_call_count: int | None,
        expected_max_tool_calls: int,
        keyword_checks: dict[str, Any],
        report_length: int,
        recursion_limit_hit: bool | None,
        has_errors: bool,
        domain_exclusion_respected: float | None,
    ) -> tuple[str, list[str]]:
        """Compute the verdict and the full list of failure reasons.

        Returns ("PASS", []) when every active gate passes, otherwise
        ("REVIEW", reasons) where reasons lists every gate that
        failed. Collecting all reasons (instead of returning on the first
        failure) lets the dashboard surface why a case was flagged without
        having to re-derive the gate logic from raw metric values.

        Format gate now also checks required_sections_present and
        citation_presence — both are deterministic and the system prompt
        mandates them, so a passing report should always score 1.0 on each.

        recursion_limit_hit is checked explicitly so the failure_reasons
        list shows "agent hit recursion limit" instead of (or alongside)
        the generic "run produced errors" entry that has_errors would
        otherwise generate. report_length enforces a minimum-length
        floor (self.min_report_length) as cheap insurance against
        truncated outputs that pass the section regex.
        """
        reasons: list[str] = []

        if has_errors:
            reasons.append("run produced errors")
        if recursion_limit_hit:
            reasons.append("agent hit recursion limit")
        if report_length < self.min_report_length:
            reasons.append(
                f"report_length {report_length} < min_report_length {self.min_report_length}"
            )

        if "judge" in self.enabled_groups:
            # no_speculation runs in both fixture and live mode and catches
            # the URL-fabrication failure mode that faithfulness only
            # partially overlaps with (faithfulness asks "is the claim grounded
            # in retrieval context"; no_speculation also rejects invented
            # citation URLs as a special case). Gating it here keeps the
            # verdict consistent with the metric's documented pass condition.
            #
            # delta_quality is fixture-only (the live agent runs cold-start
            # without history_context, so the metric isn't computed and its
            # score is None — the is not None check below skips the gate
            # automatically in live mode). It's included because correctly
            # attributing new-vs-old delta is the unique value prop of this CI
            # agent, not a nice-to-have — letting a fixture case score 0.1 on
            # delta_quality and still PASS would defeat the harness's purpose.
            # significance_justification is intentionally NOT gated:
            # significance_score_valid and score_in_expected_range
            # already cover whether the score is well-formed and within the
            # expected range, so this judge mostly adds noise on top.
            for key in ("faithfulness", "answer_relevancy", "no_speculation", "delta_quality"):
                score = judge_metrics.get(key, {}).get("score")
                if score is not None and score < self.threshold:
                    reasons.append(f"{key} {score:.2f} < threshold {self.threshold}")

        if "format" in self.enabled_groups:
            if significance_score_valid is not None and significance_score_valid < 1.0:
                reasons.append("significance_score missing or out of [1, 10]")
            if required_sections_present is not None and required_sections_present < 1.0:
                reasons.append(f"required_sections_present {required_sections_present:.2f} < 1.0")
            if citation_presence is not None and citation_presence < 1.0:
                reasons.append(f"citation_presence {citation_presence:.2f} < 1.0")

        if "routing" in self.enabled_groups:
            if directive_compliance is not None and directive_compliance < 1.0:
                reasons.append("DIRECTIVE was not obeyed (wrong sub-agent called)")
            if tool_call_count is not None and tool_call_count > expected_max_tool_calls:
                reasons.append(
                    f"tool_call_count {tool_call_count} > expected_max_tool_calls {expected_max_tool_calls}"
                )

        # score_in_expected_range is None in live mode (the case's
        # expected_significance_range encodes a synthetic scenario the
        # live report can't be expected to match) — skip the gate then.
        if score_in_expected_range is not None and score_in_expected_range < 1.0:
            reasons.append("significance_score outside expected range for this scenario")
        # Symmetric with the L1 excluded-domain gate, applied here to the URLs
        # the supervisor cited in final_report rather than to raw search
        # output (raw output is L1's responsibility). Catches cases where the
        # supervisor surfaces a URL pointing at EXCLUDED_DOMAINS in Key
        # Findings or Sources. Vacuously 1.0 when the report has no URLs or
        # when EXCLUDED_DOMAINS is empty, so the gate only fires on a real
        # violation.
        if domain_exclusion_respected is not None and domain_exclusion_respected < 1.0:
            reasons.append("excluded domain present in report citations")
        if keyword_checks.get("disallowed_keyword_hits", 0) > 0:
            reasons.append(f"disallowed_keyword_hits = {keyword_checks['disallowed_keyword_hits']}")
        hit_rate = keyword_checks.get("required_keyword_hit_rate", 1.0)
        if hit_rate < self.required_keyword_threshold:
            reasons.append(
                f"required_keyword_hit_rate {hit_rate:.2f} < threshold {self.required_keyword_threshold}"
            )

        return ("PASS" if not reasons else "REVIEW", reasons)


    async def evaluate_l2_case(self, case: L2EvalCase) -> dict[str, Any]:
        """Run a single Layer 2 eval case and return the result dict.

        All non-fatal diagnostics (supervisor-name mismatches, agent
        crashes) are appended to a single per-case errors list that
        flows into the JSON report and the SQLite ledger. Any non-empty
        errors trips the verdict to REVIEW.

        In live mode, score_in_expected_range and expected_output
        are set to None (and the gold context is dropped) because
        they encode the synthetic scenario's predicted answer, which the
        live report cannot match. See the module docstring for the full
        live vs. fixture metric breakdown.
        """
        errors: list[str] = []

        # ── Collect data (fixture or live) ────────────────────────────────
        # None (not False) in fixture mode: the agent never ran,
        # so the question "did it hit the recursion limit?" doesn't apply.
        # Storing 0 would misleadingly count fixture rows as "ran
        # without looping" in the recursion hit-rate aggregate. The live
        # branch below overwrites this with a real bool.
        recursion_limit_hit: bool | None = None
        if self.execution_mode == "fixture":
            if not case.fixture:
                return {
                    "id": case.id, "layer": 2, "category": case.category,
                    "notes": case.notes, "question": case.query,
                    "status": "SKIP", "errors": ["No fixture for this case. Add a 'fixture' block in eval_cases_layer2.json or include 'l2_live' in RUN_VARIANTS."],
                    "judge_metrics": {}, "keyword_checks": {}, "avg_judge_score": None,
                }
            fixture         = case.fixture
            final_report    = fixture.get("final_report", "")
            tool_names      = fixture.get("tool_names", [])
            tool_output     = fixture.get("tool_output", "")
            latency         = None   # Not meaningful in fixture mode.
            agent_input_tokens  = None  # No live API call in fixture mode.
            agent_output_tokens = None


        elif self.execution_mode == "live":
            (
                final_report, tool_names, tool_output, latency,
                agent_input_tokens, agent_output_tokens,
                recursion_limit_hit, live_errors,
            ) = await self._run_agent_live(case)
            errors.extend(live_errors)

        agent_total_tokens = (
            (agent_input_tokens or 0) + (agent_output_tokens or 0)
            if agent_input_tokens is not None and agent_output_tokens is not None
            else None
        )

        # ── Deterministic metrics ─────────────────────────────────────────
        format_on  = "format"  in self.enabled_groups
        routing_on = "routing" in self.enabled_groups

        required_sections_present = compute_required_sections_present(final_report) if format_on else None
        citation_presence         = compute_citation_presence(final_report)          if format_on else None
        significance_score_valid  = compute_significance_score_valid(final_report)   if format_on else None

        # Routing metrics derive from tool_names — captured from
        # astream_events in live mode, hand-authored in the fixture in
        # fixture mode. In fixture mode they reflect what the author
        # wrote, not real agent behavior, so they're NULLed (matching
        # how latency / tokens / recursion_limit_hit are treated). The
        # verdict gates guard with is not None and skip cleanly.
        is_fixture_mode_for_routing = self.execution_mode == "fixture"
        # Filter to supervisor-level tool calls only (calls to sub-agents
        # like brave_scout / tavily_analyst) so the count reflects the
        # Stop Rule's "exactly 1" expectation. Sub-agent search calls
        # (brave_search / tavily_search) also surface in astream_events
        # but those are the sub-agent's ReAct iterations — already
        # covered by recursion_limit_hit — and would inflate this
        # count for reasons unrelated to supervisor behavior.
        supervisor_tool_names = [n for n in tool_names if n in set(_ENGINE_TO_TOOL_NAME.values())]
        tool_call_count = (
            None if is_fixture_mode_for_routing
            else (compute_tool_call_count(supervisor_tool_names) if routing_on else None)
        )
        # Use case.engine (validated by Pydantic) instead of
        # substring-matching the query: tool names can legitimately
        # appear inside a query for reasons unrelated to the DIRECTIVE.
        directive_compliance = (
            None if is_fixture_mode_for_routing
            else (
                compute_directive_compliance(_ENGINE_TO_TOOL_NAME.get(case.engine), tool_names)
                if routing_on else None
            )
        )

        # Always-on diagnostic: char count of the final report. Cheap
        # signal for truncated / empty outputs that pass the section
        # regex but are otherwise garbage.
        report_length = compute_report_length(final_report)

        # Excluded-domain check on URLs cited in the report itself.
        # Live-only: in fixture mode the report is hand-authored against
        # a potentially out-of-date EXCLUDED_DOMAINS config, so the
        # gate would test fixture authorship rather than agent behavior.
        # Same NULL-in-fixture pattern as directive_compliance and
        # tool_call_count. In live mode this adds symmetric coverage
        # at the supervisor layer (L1 already tests the backend
        # filters): catches cases where the agent surfaces an
        # EXCLUDED_DOMAINS URL even when the backend was clean. Empty
        # report → empty URL list → vacuously 1.0.
        if self.execution_mode == "live":
            report_urls = extract_urls_from_text(final_report)
            domain_exclusion_respected = compute_domain_exclusion_respected(report_urls, EXCLUDED_DOMAINS)
        else:
            domain_exclusion_respected = None

        # Both delta-extraction fields are fixture-only. The score
        # encodes the agent's judgment of the new-vs-old delta, which
        # collapses without prior state — recording it in live mode
        # would yield a number whose meaning differs from the fixture
        # case. NULL in live mode is consistent with how
        # delta_quality, significance_justification, and
        # score_in_expected_range are treated.
        is_fixture_mode = self.execution_mode == "fixture"
        significance_score = (
            compute_significance_score(final_report) if is_fixture_mode else None
        )
        score_in_expected_range = (
            compute_score_in_expected_range(final_report, case.expected_significance_range)
            if is_fixture_mode else None
        )
        keyword_checks           = compute_keyword_checks(final_report, case.required_keywords, case.disallowed_keywords)


        # ── LLM judge metrics ─────────────────────────────────────────────
        # In live mode the synthetic expected_output and gold context are
        # dropped: only delta_quality reads CONTEXT, and it's already been
        # filtered out of the live judge panel; no live-safe metric reads
        # EXPECTED_OUTPUT, so passing it would be dead weight at best.
        expected_output = build_expected_output(case) if is_fixture_mode else None
        gold_context    = build_gold_context(case)    if is_fixture_mode else None
        test_case = LLMTestCase(
            input=case.query,
            actual_output=final_report or "",
            expected_output=expected_output,
            context=gold_context,
            retrieval_context=build_retrieval_context(tool_output),
        )
        judge_metrics, judge_input_tokens, judge_output_tokens = await self._run_judge_metrics(test_case)
        judge_total_tokens = (
            judge_input_tokens + judge_output_tokens
            if judge_input_tokens is not None and judge_output_tokens is not None
            else None
        )

        judge_scores = [
            judge_metrics_dict.get("score")
            for judge_metrics_dict in judge_metrics.values()
            if judge_metrics_dict.get("score") is not None
        ]
        avg_judge_score = safe_mean(judge_scores, precision_for(_FMTS["avg_judge_score"])) if judge_scores else None


        # ── Verdict ───────────────────────────────────────────────────────
        status, failure_reasons = self._compute_status(
            judge_metrics=judge_metrics,
            required_sections_present=required_sections_present,
            citation_presence=citation_presence,
            significance_score_valid=significance_score_valid,
            score_in_expected_range=score_in_expected_range,
            directive_compliance=directive_compliance,
            tool_call_count=tool_call_count,
            expected_max_tool_calls=case.expected_max_tool_calls,
            keyword_checks=keyword_checks,
            report_length=report_length,
            recursion_limit_hit=recursion_limit_hit,
            has_errors=bool(errors),
            domain_exclusion_respected=domain_exclusion_respected,
        )

        return {
            "id":                         case.id,
            "layer":                      2,
            "category":                   case.category,
            "notes":                      case.notes,
            "scenario_type":              case.scenario_type,
            "question":                   case.query,
            "expected_output":            expected_output,
            "answer":                     final_report,
            "latency_seconds":            latency,
            "agent_input_tokens":         agent_input_tokens,
            "agent_output_tokens":        agent_output_tokens,
            "agent_total_tokens":         agent_total_tokens,
            "judge_input_tokens":         judge_input_tokens,
            "judge_output_tokens":        judge_output_tokens,
            "judge_total_tokens":         judge_total_tokens,
            "required_sections_present":  required_sections_present,
            "citation_presence":          citation_presence,
            "significance_score_valid":   significance_score_valid,
            "tool_call_count":            tool_call_count,
            "directive_compliance":       directive_compliance,
            "significance_score":         significance_score,
            "score_in_expected_range":    score_in_expected_range,
            "domain_exclusion_respected": domain_exclusion_respected,
            "keyword_checks":             keyword_checks,
            "judge_metrics":              judge_metrics,
            "avg_judge_score":            avg_judge_score,
            "report_length":              report_length,
            "recursion_limit_hit":        None if recursion_limit_hit is None else (1 if recursion_limit_hit else 0),
            "status":                     status,
            "failure_reasons":            failure_reasons,
            "errors":                     errors,
            # Captured for fixture round-trip so a subsequent fixture-mode run
            # judges against the same retrieval context and tool sequence as
            # the live run that produced this record.
            "tool_output":                tool_output,
            "tool_names":                 tool_names,
        }

    async def evaluate_l2_cases(
        self,
        cases: list[L2EvalCase],
        concurrency: int = 2,
    ) -> list[dict[str, Any]]:
        """Run all Layer 2 cases concurrently, bounded by the concurrency limit."""
        semaphore = asyncio.Semaphore(max(concurrency, 1))

        async def bounded(case: L2EvalCase) -> dict[str, Any]:
            async with semaphore:
                return await self.evaluate_l2_case(case)

        return await asyncio.gather(*(bounded(case) for case in cases))


# ---------------------------------------------------------------------------
# DeepEval LLM wrapper that tracks judge token usage
# ---------------------------------------------------------------------------

class _TokenTrackingJudgeLLM(DeepEvalBaseLLM):
    """DeepEval-compatible LLM that wraps ChatOpenAI and counts tokens.

    Built so the engine can report per-case judge token costs alongside
    the existing agent token counts. One instance is created per
    evaluate_l2_case invocation and shared across every metric in the
    judge panel. After the panel runs, the engine reads
    input_tokens / output_tokens directly off the wrapper.

    ChatOpenAI is used internally (rather than the openai SDK directly)
    to match the rest of the codebase and to get standardised
    usage_metadata extraction. Schema-based generation is supported
    via with_structured_output(..., include_raw=True) so the wrapping
    AIMessage is still available for token accounting even when DeepEval
    asks for a Pydantic-typed response.

    The lock guards counter updates in a_generate because all judge
    metrics in a case run concurrently inside a single
    asyncio.gather. Without it, interleaved increments could
    corrupt totals on bursty runs. generate (sync) does NOT acquire
    the lock: asyncio.Lock cannot be used outside the running event
    loop, and DeepEval only invokes the sync path single-threadedly
    from a code path that doesn't overlap with the async path. If
    DeepEval ever starts mixing sync and async judge calls
    concurrently, switch the counter to a threading.Lock (or move
    to itertools.count) so both paths can guard safely.
    """

    def __init__(self, model: str) -> None:
        self._model_name = model
        self._client     = ChatOpenAI(model=model)
        self.input_tokens  = 0
        self.output_tokens = 0
        self._lock = asyncio.Lock()

    def load_model(self) -> ChatOpenAI:
        return self._client

    def get_model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, schema: Any = None) -> Any:
        if schema is not None:
            structured = self._client.with_structured_output(schema, include_raw=True)
            result = structured.invoke(prompt)
            self._record(result.get("raw"))
            return result["parsed"]
        response = self._client.invoke(prompt)
        self._record(response)
        return response.content

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        if schema is not None:
            structured = self._client.with_structured_output(schema, include_raw=True)
            result = await structured.ainvoke(prompt)
            async with self._lock:
                self._record(result.get("raw"))
            return result["parsed"]
        response = await self._client.ainvoke(prompt)
        async with self._lock:
            self._record(response)
        return response.content

    def _record(self, message: Any) -> None:
        """Add this call's token usage to the running totals.

        usage_metadata is LangChain's standardised dict
        ({"input_tokens": int, "output_tokens": int, ...}). Returns
        early when the field is absent — true for non-OpenAI providers
        and for some LangChain corner cases where the metadata isn't
        propagated.
        """
        usage = getattr(message, "usage_metadata", None)
        if usage:
            self.input_tokens  += int(usage.get("input_tokens", 0)  or 0)
            self.output_tokens += int(usage.get("output_tokens", 0) or 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sum_message_tokens(messages: list[Any]) -> tuple[int | None, int | None]:
    """Sum input/output token counts across every AIMessage in messages.

    Reads LangChain's standardised usage_metadata attribute, populated
    by ChatOpenAI and most modern providers (returns {"input_tokens":
    int, "output_tokens": int, "total_tokens": int}). Returns
    (None, None) when no AIMessage carries the field — typically
    because a non-OpenAI provider doesn't expose it. Returning None
    instead of (0, 0) keeps "couldn't measure" distinct from
    "actually zero" so the dashboard renders a clear NOT_EVALUATED rather
    than a misleading zero. Used by Layer 1 run_l1_subagent; Layer 2
    captures tokens differently because it streams events.
    """
    saw_usage    = False
    input_total  = 0
    output_total = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        saw_usage     = True
        input_total  += int(usage.get("input_tokens", 0) or 0)
        output_total += int(usage.get("output_tokens", 0) or 0)
    if not saw_usage:
        return None, None
    return input_total, output_total
