# Evaluation Metrics Reference

This document describes the metrics produced by the Competitive Intelligence (CI) Agent evaluation harness. The harness has **three modes**, each with its own metric set:

- **Layer 1 - Sub-agent isolation mode.** Calls the Brave or Tavily sub-agent directly (bypassing the supervisor) against the cases in `eval_cases_layer1.json`. Tests structural properties of the search-tool integration: URL counts, domain rules, required/disallowed keywords, plus operational diagnostics (latency, tokens, recursion limit). The evaluation is always done live. There is no fixture mode for L1, since its purpose is to verify real API behavior.
- **Layer 2 - Fixture mode.** Runs against pre-authored synthetic fixtures in `eval_cases_layer2.json`. Each fixture pairs a hand-written `final_report` with a `history_context` (PRIOR STATE) and `expected_answer_points` to form a self-consistent scenario. The agent's LLM does NOT run, the fixture mode is pure judge-replay over the stored report. Cheap (only judge tokens, no agent tokens), reproducible, and safe to run in continuous integration (CI) (no external search APIs, no `BRAVE_API_KEY` or `TAVILY_API_KEY` required in the runner). Ideal for iterating on judge prompts and gate logic.
- **Layer 2 - Live Mode.** Calls the real supervisor → sub-agent pipeline against the `eval_cases_layer2.json` dataset. The synthetic `history_context` is stripped (the agent runs as a cold-start research task), so metrics that depend on synthetic ground truth (`delta_quality`, `significance_justification`, `score_in_expected_range`, `significance_score`) are stored as NULL. The functional metrics faithfulness, no_speculation, answer_relevancy, format gates, routing gates, and keyword checks all run normally.

For information about the per-mode verdict logic (which metrics actually flip a case from PASS to REVIEW in each mode, and which are recorded but informational) jump to [PASS/REVIEW Gates Verdict Logic](#verdict-logic) at the bottom of this document.

### Metrics at a Glance

**Layer 1 - Sub-agent isolation metrics**

| Metric | Description |
|--------|-------------|
| [Result URL Count](#l1-result-URL-count) | How many URLs the sub-agent's tool call(s) returned. |
| [Domain Inclusion](#l1-domain-inclusion) | Whether at least one URL is from the configured `INCLUDED_DOMAINS`. |
| [Domain Exclusion](#l1-domain-exclusion) | Whether ALL URLs avoid the configured `EXCLUDED_DOMAINS`. |
| [Keyword Hit Rate](#l1-keyword-hit-rate) | Fraction of `required_keywords` found in the sub-agent's response. |
| [Disallowed Keywords](#l1-disallowed-keywords) | Count of `disallowed_keywords` found in the sub-agent's response. |
| [Latency](#l1-latency) | Wall-clock seconds from invocation to final response. |
| [Tokens](#l1-tokens) | Input / output / total tokens consumed by the sub-agent's LLM. |
| [Tool Call Count](#l1-tool-call-count) | Number of tool calls the sub-agent's ReAct loop made. |
| [Recursion Hit Rate](#l1-recursion-hit-rate) | Per-case 0/1 flag for hitting LangGraph's recursion ceiling. Averaged across cases at run level. |
| [Report Length](#l1-report-length) | Character count of the sub-agent's textual response. |

**Layer 2 - Fixture mode metrics**

| Metric | Description |
|--------|-------------|
| [Sections Present](#l2fixture-sections-present) | Fraction of the 5 mandatory report sections present. |
| [Citation Presence](#l2fixture-citation-presence) | Fraction of Key Findings bullets that end with a real URL. |
| [Keyword Hit Rate](#l2fixture-keyword-hit-rate) | Fraction of `required_keywords` found in the report. |
| [Disallowed Keywords](#l2fixture-disallowed-keywords) | Count of `disallowed_keywords` found in the report. |
| [Significance Score](#l2fixture-score) | The `SIGNIFICANCE_SCORE: N` value extracted from the report. |
| [Significance Valid](#l2fixture-significance-valid) | Whether the score is present and within the 1–10 range. |
| [Significance in Range](#l2fixture-significance-in-range) | Whether the score falls inside the case's `expected_significance_range`. |
| [Report Length](#l2fixture-report-length) | Character count of the fixture's hand-authored `final_report`. |
| [Answer Relevancy](#l2fixture-answer-relevancy) | DeepEval judge: does the report directly address the query? |
| [Faithfulness](#l2fixture-faithfulness) | DeepEval judge: are claims grounded in the retrieval context? |
| [No Speculation](#l2fixture-no-speculation) | GEval judge: does the report avoid speculation and fabricated URLs? |
| [Delta Quality](#l2fixture-delta-quality) | GEval judge: does the score reflect genuine new-vs-old delta vs PRIOR STATE? |
| [Significance Justification](#l2fixture-significance-justification) | GEval judge: does the reasoning section justify the chosen score? |
| [Judge Tokens](#l2fixture-judge-tokens) | Input / output / total tokens consumed by the judge LLM panel. |

**Layer 2 - Live mode metrics**

| Metric | Description |
|--------|-------------|
| [Sections Present](#l2live-sections-present) | Fraction of the 5 mandatory report sections present. |
| [Citation Presence](#l2live-citation-presence) | Fraction of Key Findings bullets that end with a real URL. |
| [Domain Exclusion](#l2live-domain-exclusion) | Whether ALL URLs cited in the supervisor's `final_report` avoid `EXCLUDED_DOMAINS`. |
| [Keyword Hit Rate](#l2live-keyword-hit-rate) | Fraction of `required_keywords` found in the report. |
| [Disallowed Keywords](#l2live-disallowed-keywords) | Count of `disallowed_keywords` found in the report. |
| [Directive Compliance](#l2live-directive-compliance) | Whether the supervisor called the sub-agent named in the DIRECTIVE prefix. |
| [Tool Call Count](#l2live-tool-call-count) | Number of tool calls the supervisor made (Stop Rule expects 1). |
| [Recursion Hit Rate](#l2live-recursion-hit-rate) | Per-case 0/1 flag for hitting LangGraph's recursion ceiling; averaged across cases at run level. |
| [Report Length](#l2live-report-length) | Character count of the supervisor's generated `final_report`. |
| [Latency](#l2live-latency) | Wall-clock seconds from invocation to final report. |
| [Tokens](#l2live-tokens) | Input / output / total tokens consumed by the agent LLM (supervisor + sub-agent). |
| [Answer Relevancy](#l2live-answer-relevancy) | DeepEval judge: does the report directly address the query? |
| [Faithfulness](#l2live-faithfulness) | DeepEval judge: are claims grounded in the retrieval context? |
| [No Speculation](#l2live-no-speculation) | GEval judge: does the report avoid speculation and fabricated URLs? |
| [Judge Tokens](#l2live-judge-tokens) | Input / output / total tokens consumed by the judge LLM panel. |


**Summary metrics**

| Metric | Description |
|--------|-------------|
| [Pass Rate](#pass-rate) | Fraction of cases in the run that ended with status PASS. |
| [Avg Judge Score](#avg-judge-score) | Run-level mean of per-case `avg_judge_score` (which is itself the mean of the judge panel scores per case). |

---

## Layer 1 - Sub-agent isolation metrics

Layer 1 evaluates each search sub-agent (`brave_agent` or `tavily_agent`) in isolation by calling it directly, bypassing the supervisor. The query goes straight to the sub-agent (selected per case via `engine: brave|tavily`), and the metrics measure structural properties of what came back (URL counts, domain compliance, keyword presence) plus operational diagnostics (latency, tokens, tool calls, recursion). No LLM judges run at this layer. Sub-agent reasoning quality is judged at layer 2.

---

<a id="l1-result-URL-count"></a>
### Result URL Count

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 |
| Direction | higher is better |
| Toggle group | always-on (sub-agent group) |
| Stored as | `eval_cases.result_count` |
| Computed by | `compute_result_URL_count` in `eval_metrics.py` |
| Pass condition | `value >= case.expected_min_results` |

**What it evaluates:** The number of distinct URLs the sub-agent's tool call(s) returned for the case's query.

**Why it is included:** A count of 0 means the API succeeded but returned nothing. Could be a malformed query, an over-aggressive domain filter, or a quota issue. Comparing against the case's `expected_min_results` catches these silent failures. Multi-call ReAct retries are reflected because every `ToolMessage` in the response is collected, not just the first.

**How it is calculated:** Every `ToolMessage` is concatenated into a `tool_output` blob, then `extract_urls_from_text` regex-extracts all URLs. The metric is `len(result_urls)`.

---

<a id="l1-domain-inclusion"></a>
### Domain Inclusion

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | always-on (sub-agent group) |
| Stored as | `eval_cases.domain_inclusion_present` |
| Computed by | `compute_domain_inclusion_present` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** Whether at least one URL from the project-wide `INCLUDED_DOMAINS` list (in `config.py`) appears in the sub-agent's results.

**Why it is included:** Some research scenarios require results from specific authoritative sources. If `INCLUDED_DOMAINS` is configured, this metric verifies the sub-agent is actually hitting them. Vacuously 1.0 when `INCLUDED_DOMAINS` is empty, so it never triggers a false negative on cases where no inclusion list applies.

**How it is calculated:** Returns 1.0 if any URL contains any `INCLUDED_DOMAINS` entry as a substring (or if the list is empty), 0.0 otherwise.

---

<a id="l1-domain-exclusion"></a>
### Domain Exclusion

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | always-on (sub-agent group) |
| Stored as | `eval_cases.domain_exclusion_respected` |
| Computed by | `compute_domain_exclusion_respected` in `eval_metrics.py` |
| Pass condition | `value == 1.0` |

**What it evaluates:** Whether ALL URLs avoid the `EXCLUDED_DOMAINS` list (e.g., `linkedin.com`, job boards, social media) configured in `config.py`.

**Why it is included:** Excluding low-signal sources is a real production requirement for competitive intelligence. This gate catches cases where the sub-agent's domain filter isn't being honored by the search backend. It is binary by design, so even one excluded domain in the result set is a failure.

**How it is calculated:** Returns 1.0 if no URL contains any `EXCLUDED_DOMAINS` entry as a substring, 0.0 if any URL matches.

---

<a id="l1-keyword-hit-rate"></a>
### Keyword Hit Rate

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always-on (keyword composite) |
| Stored as | `eval_cases.required_keyword_hit_rate` |
| Computed by | `compute_keyword_checks` in `eval_metrics.py` |
| Pass condition | `value >= REQUIRED_KEYWORD_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** The fraction of the case's `required_keywords` list that appears anywhere in the sub-agent's textual `response_text`.

**Why it is included:** Required keywords are an authoring affordance for "this answer must mention X", e.g., a specific competitor, product, or term that any reasonable response should reference. The threshold (configurable via `REQUIRED_KEYWORD_THRESHOLD`) lets you tolerate partial coverage rather than demanding all-or-nothing.

**How it is calculated:** Each required keyword is normalized (lowercased, accent-stripped via `normalize_text`) and substring-matched against the normalized response. Hit rate = matches / total required.

---

<a id="l1-disallowed-keywords"></a>
### Disallowed Keywords

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 |
| Direction | lower is better (target: 0) |
| Toggle group | always-on (keyword composite) |
| Stored as | `eval_cases.disallowed_keyword_hits` |
| Computed by | `compute_keyword_checks` in `eval_metrics.py` |
| Pass condition | `value == 0` |

**What it evaluates:** The count of `disallowed_keywords` (e.g., refusal phrases like "I cannot", "I don't have access") found in the sub-agent's response.

**Why it is included:** Catches the failure mode where the sub-agent returns a refusal or hedging phrase instead of actual content, typically because the search backend returned an error and the LLM defaulted to apologizing. It is binary-by-design, so even one hit fails the gate.

**How it is calculated:** Each disallowed keyword is normalized and substring-matched against the normalized response. The metric is the count of distinct disallowed keywords that matched.

---

<a id="l1-latency"></a>
### Latency

| Field | Value |
|-------|-------|
| Range | float seconds ≥ 0 |
| Direction | lower is better |
| Toggle group | always-on (latency group) |
| Stored as | `eval_cases.latency_seconds`, `eval_runs.avg_latency_seconds` |
| Computed by | `time.perf_counter()` inside `run_l1_subagent` |
| Pass condition | (not gated) |

**What it evaluates:** Wall-clock time from invoking the sub-agent to receiving its final response.

**Why it is included:** Search-API integration latency is a production concern. Trend monitoring catches API slowdowns and caching regressions before users notice.

**How it is calculated:** `round(time.perf_counter() - started, precision_for(_FMTS["latency_seconds"]))` around the `await target_agent.ainvoke(...)` call. Stored precision is derived from the metric's registry `fmt` so changing the display precision in `eval_metric_registry.py` updates storage in lockstep.

---

<a id="l1-tokens"></a>
### Tokens

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 (each of input / output / total) |
| Direction | lower is better |
| Toggle group | always-on (tokens group) |
| Stored as | `eval_cases.agent_input_tokens` / `agent_output_tokens` / `agent_total_tokens`, with run-level averages on `eval_runs` |
| Computed by | `_sum_message_tokens` in `eval_engine.py` (reads LangChain's `usage_metadata`) |
| Pass condition | (not gated) |

**What it evaluates:** Input, output, and total tokens consumed by the sub-agent's LLM calls during one case.

**Why it is included:** Token usage is the dominant cost driver. Tracking per case lets you compare provider/model changes and catch regressions where prompt edits inflate consumption.

**How it is calculated:** Iterates every `AIMessage` in the sub-agent's returned messages, reads `usage_metadata`, and sums `input_tokens` / `output_tokens`. Total = input + output. Returns `None` (not zero) when no message carried `usage_metadata` (e.g., a non-OpenAI provider) so the dashboard renders "not evaluated" rather than a misleading zero.

---

<a id="l1-tool-call-count"></a>
### Tool Call Count

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 |
| Direction | informational (no fixed direction; depends on case) |
| Toggle group | routing (but L1 computes it always, since L1 has no routing toggle) |
| Stored as | `eval_cases.tool_call_count`, `eval_runs.avg_tool_call_count` |
| Computed by | `sum(1 for m in messages if isinstance(m, ToolMessage))` in `run_l1_subagent` |
| Pass condition | (not gated for L1) |

**What it evaluates:** How many times the sub-agent's ReAct loop invoked the search tool during one case.

**Why it is included:** Sub-agents are ReAct loops that can call the tool multiple times for refinement. A spike in average call count signals the agent is iterating excessively, raising a soft cost concern even if the recursion limit isn't hit.

**How it is calculated:** Direct count of `ToolMessage` instances in the agent's returned messages list. Counted directly (not via `extract_all_tool_outputs`) so empty-content tool responses still count as real calls.

---

<a id="l1-recursion-hit-rate"></a>
### Recursion Hit Rate

| Field | Value |
|-------|-------|
| Range | 0 or 1 per case; 0.0 – 1.0 at run level (average of cases) |
| Direction | lower is better (target: 0) |
| Toggle group | always-on (diagnostic group) |
| Stored as | `eval_cases.recursion_limit_hit` (per case), `eval_runs.avg_recursion_limit_hit_rate` (run level) |
| Computed by | `type(exc).__name__ == "GraphRecursionError"` in `run_l1_subagent` exception handler |
| Pass condition | `value == 0` (gate adds "agent hit recursion limit" to failure_reasons) |

**What it evaluates:** A 0/1 flag for whether LangGraph's `GraphRecursionError` fired. It evaluates if the sub-agent's ReAct loop ran past LangGraph's super-step ceiling (counting every node execution, each LLM turn AND each tool call).

**Why it is included:** Distinct from a generic crash, a recursion-limit hit means the agent didn't converge on a stop condition, which is a behavior bug rather than an integration error. Tracking the rate over time answers the question: Are recent runs looping more often after my last prompt edit?

**How it is calculated:** The L1 invocation is wrapped in `try / except Exception`. On exception, `recursion_limit_hit = type(exc).__name__ == "GraphRecursionError"`. The 0/1 per case is mean-aggregated across the run via `safe_mean`, with NULL fixture rows excluded.

---

<a id="l1-report-length"></a>
### Report Length

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 (characters) |
| Direction | (no fixed direction; gated against a floor) |
| Toggle group | always-on (diagnostic group) |
| Stored as | `eval_cases.report_length` |
| Computed by | `compute_report_length(response_text)` in `eval_metrics.py` |
| Pass condition | `value >= MIN_REPORT_LENGTH` (configurable in `eval_config.py`) |

**What it evaluates:** Character count of the sub-agent's textual `response_text` (the LLM-generated commentary around the search results. It is distinct from `result_count`, which counts URLs).

**Why it is included:** Catches truncated/empty outputs that pass other gates but are obviously broken. A 30-character "response" likely indicates the LLM bailed mid-generation or returned a refusal stub.

**How it is calculated:** `len(response_text)`.

---

## Layer 2 - Fixture mode

Fixture mode runs the full layer 2 metric panel against hand-authored test data. The fixture's `final_report`, `tool_output`, and `tool_names` are read directly from `eval_cases_layer2.json` and injected into the judge pipeline, so the agent's LLM never runs. This mode is for testing the metrics themselves, the judges' prompts, and the format/routing gates against synthetic scenarios with known ground truth (`history_context` + `expected_answer_points`). Free, deterministic, continuous integration-safe.

Because no agent runs, the operational metrics are NULL: `latency_seconds`, `agent_*_tokens`, `recursion_limit_hit`, `tool_call_count`, `directive_compliance`, `domain_exclusion_respected`. The fixture's `tool_names` field is preserved for inspection but no longer feeds verdict gates.

---

<a id="l2fixture-sections-present"></a>
### Sections Present

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | format |
| Stored as | `eval_cases.required_sections_present` |
| Computed by | `compute_required_sections_present` in `eval_metrics.py` |
| Pass condition | `value == 1.0` |

**What it evaluates:** The fraction of the 5 mandatory report sections present in the fixture's `final_report`: Executive Summary, Key Findings, Competitive Implication, Sources, Significance Score Reasoning.

**Why it is included:** The system prompt mandates this structure. A passing report should always score 1.0. Anything less means the report skipped a required section, which downstream consumers (dashboards, alerts) rely on.

**How it is calculated:** Regex-matches each required section header (case-insensitive `## Section Name`) against the report text. Score = matched / total.

---

<a id="l2fixture-citation-presence"></a>
### Citation Presence

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | format |
| Stored as | `eval_cases.citation_presence` |
| Computed by | `compute_citation_presence` in `eval_metrics.py` |
| Pass condition | `value == 1.0` |

**What it evaluates:** The fraction of Key Findings bullets that end with a real URL in parentheses (the citation convention the system prompt enforces).

**Why it is included:** Bullets without source URLs are claims the user can't verify. Format-gated to 1.0 because every finding should be traceable.

**How it is calculated:** Extracts the Key Findings section, splits into bullets, regex-checks each for a trailing `(http...)` URL. Score = bullets-with-URL / total-bullets.

---

<a id="l2fixture-keyword-hit-rate"></a>
### Keyword Hit Rate

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always-on (keyword composite) |
| Stored as | `eval_cases.required_keyword_hit_rate`, `eval_runs.avg_required_keyword_hit_rate` |
| Computed by | `compute_keyword_checks` in `eval_metrics.py` |
| Pass condition | `value >= REQUIRED_KEYWORD_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** Fraction of the case's `required_keywords` list that appears in the fixture's `final_report`.

**Why it is included:** Same as for L1, required keywords assert "this report must mention X" (e.g., a specific competitor, term, score label like `SIGNIFICANCE_SCORE`). The threshold is configurable to allow partial credit.

**How it is calculated:** Identical to L1: Each required keyword is normalized and substring-matched against the normalized report. Hit rate = matches / total required.

---

<a id="l2fixture-disallowed-keywords"></a>
### Disallowed Keywords

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 |
| Direction | lower is better (target: 0) |
| Toggle group | always-on (keyword composite) |
| Stored as | `eval_cases.disallowed_keyword_hits`, `eval_runs.avg_disallowed_keyword_hits` |
| Computed by | `compute_keyword_checks` in `eval_metrics.py` |
| Pass condition | `value == 0` |

**What it evaluates:** Count of `disallowed_keywords` (refusal phrases, hedge words) found in the fixture's `final_report`.

**Why it is included:** Catches reports that defaulted to refusal/apology language which is a content-quality red flag. It is binary-by-design.

**How it is calculated:** Same as L1: Normalized substring matching, the metric is the count of distinct disallowed keywords matched.

---

<a id="l2fixture-score"></a>
### Significance Score

| Field | Value |
|-------|-------|
| Range | integer 1 – 10 (or NULL if absent/invalid) |
| Direction | (no fixed direction — depends on scenario) |
| Toggle group | always-on (delta group) |
| Stored as | `eval_cases.significance_score` |
| Computed by | `compute_significance_score` in `eval_metrics.py` |
| Pass condition | (not gated directly, it feeds into Significance in Range) |

**What it evaluates:** The integer value extracted from the report's `SIGNIFICANCE_SCORE: X` line, where the agent encodes how strategically significant this round's findings are relative to PRIOR STATE.

**Why it is included:** Recorded as a diagnostic so you can see the score distribution. The score's meaning collapses without prior state, which is why it's NULL in Live mode. In fixture mode the synthetic `history_context` makes it meaningful.

**How it is calculated:** Regex-matches `SIGNIFICANCE_SCORE:\s*(\d+)` against the report and parses the captured integer. Returns `None` if not found.

---

<a id="l2fixture-significance-valid"></a>
### Significance Valid

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | format |
| Stored as | `eval_cases.significance_score_valid` |
| Computed by | `compute_significance_score_valid` in `eval_metrics.py` |
| Pass condition | `value == 1.0` |

**What it evaluates:** Whether `SIGNIFICANCE_SCORE: X` appears in the report with X parseable as an integer in range 1-10.

**Why it is included:** Catches three failure modes the agent can drift into: A malformed score line (e.g., `SIGNIFICANCE: high` instead of `SIGNIFICANCE_SCORE: 7`), a missing score, or an out-of-range integer. Acts as a precondition for the other score-dependent metrics (`significance_score`, `score_in_expected_range`, `significance_justification`), which silently go NULL when the line can't be parsed. Without this gate, you'd see the downstream NULLs but not know they all stem from the same upstream format break. It also matters because the score is a contractual output read by downstream automation (alerts, dashboards).

**How it is calculated:** 1.0 if `compute_significance_score` returns a value in [1, 10], 0.0 otherwise (including missing).

---

<a id="l2fixture-significance-in-range"></a>
### Significance in Range

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | always-on (delta group) |
| Stored as | `eval_cases.score_in_expected_range` |
| Computed by | `compute_score_in_expected_range` in `eval_metrics.py` |
| Pass condition | `value == 1.0` |

**What it evaluates:** Whether the extracted `significance_score` falls inside the case's `expected_significance_range` (e.g., `[1, 3]` for a `no_change` scenario, `[7, 10]` for a `major_new_entity` scenario).

**Why it is included:** The synthetic scenario type defines what a "correct" delta verdict looks like. This gate catches the agent inflating an obvious non-event to 8 or burying a genuine new entity at 2.

**How it is calculated:** 1.0 if `expected_min <= significance_score <= expected_max`, 0.0 otherwise (including when the score is missing or invalid).

---

<a id="l2fixture-report-length"></a>
### Report Length

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 (characters) |
| Direction | (no fixed direction; gated against a floor) |
| Toggle group | always-on (diagnostic group) |
| Stored as | `eval_cases.report_length` |
| Computed by | `compute_report_length(final_report)` in `eval_metrics.py` |
| Pass condition | `value >= MIN_REPORT_LENGTH` (configurable in `eval_config.py`) |

**What it evaluates:** Character count of the fixture's hand-authored `final_report`.

**Why it is included:** Doubles as a fixture-authorship sanity check. It catches the case where someone pasted in a stub fixture by mistake. Same gate threshold as L1 / L2 live so the rule is uniform.

**How it is calculated:** `len(final_report)`.

---

<a id="l2fixture-answer-relevancy"></a>
### Answer Relevancy

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.answer_relevancy` |
| Computed by | DeepEval `AnswerRelevancyMetric` |
| Pass condition | `score >= JUDGE_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** Whether the report directly addresses the monitoring query (the `INPUT` of the test case).

**Why it is included:** Catches reports that are well-formatted and faithful but off-topic (e.g., a Nike-Europe query that returned an Adidas-Asia report). A standard DeepEval check.

**How it is calculated:** DeepEval's built-in metric extracts statements from the report and asks the judge LLM to rate each for relevance to the input query. Score = relevant / total statements.

---

<a id="l2fixture-faithfulness"></a>
### Faithfulness

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.faithfulness` |
| Computed by | DeepEval `FaithfulnessMetric` |
| Pass condition | `score >= JUDGE_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** Whether all factual claims in the report are grounded in the `retrieval_context` (the search-tool output stored in the fixture's `tool_output`).

**Why it is included:** A primary hallucination check. It fails the report when claims aren't supported by what the tool actually returned. Complementary to (and overlaps with) `no_speculation`.

**How it is calculated:** DeepEval extracts claims from the report and asks the judge LLM to verify each against the retrieval context. Score = supported / total claims.

---

<a id="l2fixture-no-speculation"></a>
### No Speculation

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.no_speculation` |
| Computed by | Custom `GEval` in `eval_engine.build_judge_metrics` |
| Pass condition | `score >= JUDGE_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** Whether the report avoids speculating beyond the search results, including specifically not fabricating placeholder URLs when a real source is unavailable.

**Why it is included:** Faithfulness catches plausible-sounding but unsupported claims, while this metric catches the related failure where the agent invents a citation to make a claim look sourced. Reads `RETRIEVAL_CONTEXT` so it works in both fixture and live mode.

**How it is calculated:** Single GEval call with criteria: "Every factual claim in Key Findings must be traceable to the search results. If a URL is not available for a fact, the report should omit it rather than fabricate a placeholder URL or invent a source." The judge LLM scores 0.0–1.0.

---

<a id="l2fixture-delta-quality"></a>
### Delta Quality

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.delta_quality` |
| Computed by | Custom `GEval` in `eval_engine.build_judge_metrics` |
| Pass condition | `score >= JUDGE_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** Whether the report correctly attributes findings as new vs. already-known relative to the synthetic `history_context` (PRIOR STATE) injected via the gold context.

**Why it is included:** Delta reasoning is the unique requirement of this CI agent. The score must reflect only the truly new delta, not the absolute importance of pre-existing information. Penalises both presenting known items as new AND dismissing genuinely new findings as known.

**How it is calculated:** Single GEval call reading `INPUT`, `ACTUAL_OUTPUT`, `EXPECTED_OUTPUT`, and `CONTEXT`. The CONTEXT contains the case's `history_context` formatted as PRIOR STATE. Judge LLM scores 0.0–1.0. **Fixture-only**. Dropped from the live judge panel because the live agent gets no `history_context` (cold-start).

---

<a id="l2fixture-significance-justification"></a>
### Significance Justification

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.significance_justification` |
| Computed by | Custom `GEval` in `eval_engine.build_judge_metrics` |
| Pass condition | (not gated; computed and recorded but doesn't flip the verdict) |

**What it evaluates:** Whether the SIGNIFICANCE_SCORE assigned is logically consistent with the report's own *Significance Score Reasoning* section and *Key Findings*. A score of 7+ requires a brand-new strategic entity, 1–2 requires that nothing meaningfully new was found.

**Why it is included:** A trend-monitoring signal that pairs with `delta_quality`: even if the delta judgment is right, the agent's stated reasoning should support the chosen number. Unlike `delta_quality`, this metric is **not** verdict-flipping — `significance_score_valid` and `score_in_expected_range` already gate whether the score is well-formed and within the expected range, so adding this judge to `_compute_status` would mostly contribute noise. The score is still recorded on the case row and shown on the dashboard for trend analysis (see *PASS/REVIEW Gates Verdict Logic* for the gating split).

**How it is calculated:** Single GEval call reading `INPUT` and `ACTUAL_OUTPUT` (does NOT mechanically read CONTEXT, but the criteria text references prior state). **Fixture-only:** In cold-start live mode every finding is trivially "new" so the metric collapses to rubber-stamping high scores.

---

<a id="l2fixture-judge-tokens"></a>
### Judge Tokens

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 (each of input / output / total) |
| Direction | lower is better |
| Toggle group | always-on (judge_tokens group; populated whenever the judge panel runs) |
| Stored as | `eval_cases.judge_input_tokens` / `judge_output_tokens` / `judge_total_tokens`, with run-level averages on `eval_runs` |
| Computed by | `_TokenTrackingJudgeLLM` wrapper in `eval_engine.py` |
| Pass condition | (not gated) |

**What it evaluates:** Input, output, and total tokens consumed by the entire judge LLM panel for one case (every judge metric in the registry × 1+ judge calls each).

**Why it is included:** In fixture mode, the judge panel is the only thing burning OpenAI budget, agent tokens are NULL. This metric makes that cost visible. Useful for spotting prompt-bloat regressions (e.g., adding a long criterion string to a GEval and watching tokens jump).

**How it is calculated:** A `_TokenTrackingJudgeLLM` (subclass of `DeepEvalBaseLLM` wrapping `ChatOpenAI`) is instantiated per case and shared across every metric in the panel. Its `generate` / `a_generate` methods read `usage_metadata` from each LangChain response and accumulate counts. Total = input + output.

---

## Layer 2 - Live mode

Live mode calls the real supervisor → sub-agent pipeline against the layer 2 dataset. Used periodically for end-to-end regression checking against the actual Brave and Tavily APIs. The synthetic `history_context` is stripped (the agent runs as a cold-start research task), so metrics that encode delta vs. prior state are not meaningful and are stored as NULL: `delta_quality`, `significance_justification`, `score_in_expected_range`, `significance_score`. Functional metrics including faithfulness, no_speculation, answer_relevancy, format gates, routing gates, and keyword checks all apply normally.

Because the agent actually runs, operational metrics (latency, agent tokens, tool_call_count, recursion_limit_hit, directive_compliance) all carry real measured values.

---

<a id="l2live-sections-present"></a>
### Sections Present

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | format |
| Stored as | `eval_cases.required_sections_present` |
| Computed by | `compute_required_sections_present` in `eval_metrics.py` |
| Pass condition | `value == 1.0` |

**What it evaluates:** Same as for L2 fixture mode: Fraction of the 5 mandatory sections (Executive Summary, Key Findings, Competitive Implication, Sources, Significance Score Reasoning) present in the supervisor's generated `final_report`.

**Why it is included:** The system prompt mandates this structure. A missing section here is a genuine agent-behavior regression (the supervisor failed to follow its template).

**How it is calculated:** Identical to L2 fixture mode: Regex-matches each required section header against the report text. Score = matched / total.

---

<a id="l2live-citation-presence"></a>
### Citation Presence

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | format |
| Stored as | `eval_cases.citation_presence` |
| Computed by | `compute_citation_presence` in `eval_metrics.py` |
| Pass condition | `value == 1.0` |

**What it evaluates:** Same as for L2 fixture mode: Fraction of Key Findings bullets that end with a real URL in parentheses.

**Why it is included:** Identical to L2 fixture mode. The system prompt requires every finding to be traceable. The gate enforces the contract.

**How it is calculated:** Same as for L2 fixture mode.

---

<a id="l2live-domain-exclusion"></a>
### Domain Exclusion

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 (NULL in L2 fixture mode) |
| Direction | higher is better |
| Toggle group | always-on (sub-agent group) |
| Stored as | `eval_cases.domain_exclusion_respected` |
| Computed by | `compute_domain_exclusion_respected` in `eval_metrics.py` (called against URLs extracted from `final_report`) |
| Pass condition | `value == 1.0` (gate adds "excluded domain present in report citations" to failure_reasons) |

**What it evaluates:** Whether ALL URLs cited in the supervisor's `final_report` (Key Findings + Sources) avoid the `EXCLUDED_DOMAINS` list configured in the agent's `config.py`. Symmetric with the L1 excluded-domain gate, but applied to what the supervisor surfaced rather than what the search backend returned.

**Why it is included:** Catches the failure mode where the search backend was clean but the supervisor (or sub-agent) still cited an excluded domain in its output. For example, surfacing a `linkedin.com` URL from a tool result the search filter let through, or paraphrasing in an excluded source from training data. Without this gate, an excluded URL appearing in the report's Key Findings or Sources would slip past the L1 backend check (which doesn't run for L2 cases) and through `citation_presence` (which only checks that URLs are present, not which domain). Live-only because in L2 fixture mode the hand-authored `final_report` is written against a possibly historical `EXCLUDED_DOMAINS` config. Running this gate there would test fixture authorship rather than agent behavior, so it's stored as NULL.

**How it is calculated:** URLs are extracted from the live `final_report` via `extract_urls_from_text`, then passed to `compute_domain_exclusion_respected(report_urls, EXCLUDED_DOMAINS)`. Returns 1.0 if no URL contains any `EXCLUDED_DOMAINS` entry as a substring, 0.0 if any URL matches. Vacuously 1.0 when `EXCLUDED_DOMAINS` is empty or the report has no URLs.

---

<a id="l2live-keyword-hit-rate"></a>
### Keyword Hit Rate

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always-on (keyword composite) |
| Stored as | `eval_cases.required_keyword_hit_rate`, `eval_runs.avg_required_keyword_hit_rate` |
| Computed by | `compute_keyword_checks` in `eval_metrics.py` |
| Pass condition | `value >= REQUIRED_KEYWORD_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** Same as for L2 fixture mode: Fraction of the case's `required_keywords` found in the live supervisor's generated `final_report`.

**Why it is included:** Catches reports that drift off-topic or fail to mention contractually required terms (e.g., a cross-cutting `SIGNIFICANCE_SCORE` label).

**How it is calculated:** Same as L1 / L2 fixture mode: Normalized substring matching, rate = matches / total.

---

<a id="l2live-disallowed-keywords"></a>
### Disallowed Keywords

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 |
| Direction | lower is better (target: 0) |
| Toggle group | always-on (keyword composite) |
| Stored as | `eval_cases.disallowed_keyword_hits`, `eval_runs.avg_disallowed_keyword_hits` |
| Computed by | `compute_keyword_checks` in `eval_metrics.py` |
| Pass condition | `value == 0` |

**What it evaluates:** Same as for L2 fixture mode: Count of refusal/hedge phrases in the report.

**Why it is included:** Catches the live agent reverting to apology mode when something upstream failed (e.g., the search returned no results and the supervisor gave up rather than reporting the empty state).

**How it is calculated:** Same as L1 / L2 fixture mode.

---

<a id="l2live-directive-compliance"></a>
### Directive Compliance

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | routing |
| Stored as | `eval_cases.directive_compliance` |
| Computed by | `compute_directive_compliance` in `eval_metrics.py` |
| Pass condition | `value == 1.0` |

**What it evaluates:** Whether the supervisor called the sub-agent named in the case's DIRECTIVE prefix (e.g., `DIRECTIVE: Use 'brave_scout' only.` → must call `brave_scout`, not `tavily_analyst`).

**Why it is included:** Each case targets one specific sub-agent. Routing failures (the supervisor ignoring the DIRECTIVE) are functional regressions. The test harness can't validate a Brave-specific behavior if Tavily was actually called. Live-only because the value is NULL in fixture mode (where `tool_names` is hand-authored and not behavioral).

**How it is calculated:** Maps `case.engine` ("brave" / "tavily") to the expected tool name (`brave_scout` / `tavily_analyst`), then checks whether that name appears in the live `tool_names` list captured from `astream_events`. Returns 1.0 if the expected name is present, 0.0 otherwise.

---

<a id="l2live-tool-call-count"></a>
### Tool Call Count

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 |
| Direction | lower is better (Stop Rule expects 1) |
| Toggle group | routing |
| Stored as | `eval_cases.tool_call_count`, `eval_runs.avg_tool_call_count` |
| Computed by | `compute_tool_call_count(tool_names)` in `eval_metrics.py` |
| Pass condition | `value <= case.expected_max_tool_calls` (set per case in `eval_cases_layer2.json`) |

**What it evaluates:** Number of supervisor-level tool calls made during one case — i.e., how many times the supervisor invoked a sub-agent (`brave_scout` or `tavily_analyst`). The sub-agent's own internal search-tool calls (`brave_search` / `tavily_search`) are excluded so this metric reflects supervisor behavior only.

**Why it is included:** The system prompt's Stop Rule requires the supervisor to call the sub-agent exactly once. Multiple supervisor-level calls indicate the supervisor is looping or second-guessing, a behavior regression, even if the recursion limit isn't hit. Sub-agent looping (the sub-agent re-querying the search backend many times inside its ReAct loop) is a separate concern, captured by `recursion_limit_hit` and the L1 `tool_call_count` diagnostic.

**How it is calculated:** Every `on_tool_start` event from `astream_events` appends its tool name to a list. The list is then filtered to entries whose name is a supervisor-level sub-agent tool (the values of `_ENGINE_TO_TOOL_NAME`: `brave_scout` and `tavily_analyst`); the metric is `len(filtered_list)`. Filtering happens at the call site in `_run_agent_live`, not inside `compute_tool_call_count` itself, so the full unfiltered `tool_names` list is preserved on the case row for inspection (e.g., dashboards showing every tool the agent touched).

---

<a id="l2live-recursion-hit-rate"></a>
### Recursion Hit Rate

| Field | Value |
|-------|-------|
| Range | 0 or 1 per case; 0.0 – 1.0 at run level |
| Direction | lower is better (target: 0) |
| Toggle group | always-on (diagnostic group) |
| Stored as | `eval_cases.recursion_limit_hit`, `eval_runs.avg_recursion_limit_hit_rate` |
| Computed by | `type(exc).__name__ == "GraphRecursionError"` in `_run_agent_live` exception handler |
| Pass condition | `value == 0` (gate adds "agent hit recursion limit" to failure_reasons) |

**What it evaluates:** A 0/1 flag for whether LangGraph's `GraphRecursionError` propagated out of the supervisor invocation. The supervisor and each sub-agent run as separate graphs with **independent** recursion budgets (LangGraph's super-step ceiling per graph, counting every node execution, each LLM turn AND each tool call). In practice this metric fires when the sub-agent's ReAct loop runs past its own ceiling (the supervisor itself only needs a handful of super-steps for the standard handoff pattern).

**Why it is included:** Distinct from a generic crash, a recursion-limit hit means either the sub-agent's ReAct loop didn't converge on a stop condition (the common case) or the supervisor got stuck in a handoff back-and-forth (rare, since the Stop Rule expects 1 sub-agent call). Tracking the rate over time catches regressions where prompt edits inflate looping behavior.

**How it is calculated:** The live supervisor invocation is wrapped in `try / except`. On exception, `recursion_limit_hit = type(exc).__name__ == "GraphRecursionError"`. Run-level rate = `safe_mean` of per-case values (NULL fixture rows excluded).

---

<a id="l2live-report-length"></a>
### Report Length

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 (characters) |
| Direction | (no fixed direction; gated against a floor) |
| Toggle group | always-on (diagnostic group) |
| Stored as | `eval_cases.report_length` |
| Computed by | `compute_report_length(final_report)` in `eval_metrics.py` |
| Pass condition | `value >= MIN_REPORT_LENGTH` (configurable in `eval_config.py`) |

**What it evaluates:** Character count of the supervisor's generated `final_report`.

**Why it is included:** Catches truncated outputs that pass section/citation gates but are obviously broken (e.g., the supervisor emitted a stub like `## Executive Summary\n\nSIGNIFICANCE_SCORE: 1` and stopped).

**How it is calculated:** `len(final_report)`.

---

<a id="l2live-latency"></a>
### Latency

| Field | Value |
|-------|-------|
| Range | float seconds ≥ 0 |
| Direction | lower is better |
| Toggle group | always-on (latency group) |
| Stored as | `eval_cases.latency_seconds`, `eval_runs.avg_latency_seconds` |
| Computed by | `time.perf_counter()` inside `_run_agent_live` |
| Pass condition | (not gated) |

**What it evaluates:** Wall-clock time from invoking the supervisor pipeline to emitting the final report.

**Why it is included:** End-to-end pipeline latency includes search-API time, LLM time, and supervisor coordination overhead. Trend monitoring catches regressions in any of those components.

**How it is calculated:** `round(time.perf_counter() - started, precision_for(_FMTS["latency_seconds"]))` around the supervisor's `astream_events` loop. Stored precision is derived from the metric's registry `fmt`.

---

<a id="l2live-tokens"></a>
### Tokens

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 (each of input / output / total) |
| Direction | lower is better |
| Toggle group | always-on (tokens group) |
| Stored as | `eval_cases.agent_input_tokens` / `agent_output_tokens` / `agent_total_tokens`, with run-level averages on `eval_runs` |
| Computed by | Summed from every `on_chat_model_end` event in `_run_agent_live` |
| Pass condition | (not gated) |

**What it evaluates:** Input, output, and total tokens consumed by the agent's LLM during one case. Covers both the supervisor and the sub-agent that the supervisor invoked.

**Why it is included:** Same cost-monitoring rationale as in layer 1, scoped to the full supervisor + sub-agent pipeline. Distinct from `judge_tokens`, which covers the judge LLM panel.

**How it is calculated:** During the live `astream_events` loop, every `on_chat_model_end` event's `output.usage_metadata` contributes its `input_tokens` / `output_tokens` to the running totals. Total = input + output.

---

<a id="l2live-answer-relevancy"></a>
### Answer Relevancy

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.answer_relevancy` |
| Computed by | DeepEval `AnswerRelevancyMetric` |
| Pass condition | `score >= JUDGE_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** Same as for L2 fixture mode: Whether the report directly addresses the monitoring query.

**Why it is included:** Same purpose and same metric as in L2 fixture mode: Works identically in live mode because it doesn't depend on synthetic ground truth.

**How it is calculated:** Same DeepEval algorithm as in L2 fixture mode, judged against the live-generated `final_report`.

---

<a id="l2live-faithfulness"></a>
### Faithfulness

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.faithfulness` |
| Computed by | DeepEval `FaithfulnessMetric` |
| Pass condition | `score >= JUDGE_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** Same as for L2 fixture mode: Whether the report's claims are grounded in the `retrieval_context`. However, in live mode the retrieval_context is the *real* Brave/Tavily output, not a hand-authored fixture.

**Why it is included:** The strongest hallucination signal in live mode. The agent has to be faithful to actual search results, not fictional fixtures.

**How it is calculated:** Same DeepEval algorithm as in L2 fixture mode. The retrieval context is built from the concatenated tool outputs captured during the live run.

---

<a id="l2live-no-speculation"></a>
### No Speculation

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.no_speculation` |
| Computed by | Custom `GEval` in `eval_engine.build_judge_metrics` |
| Pass condition | `score >= JUDGE_THRESHOLD` (configurable in `eval_config.py`) |

**What it evaluates:** Same as for L2 fixture mode: Whether or not the report avoids speculation and fabricated URLs, judged against the live retrieval context.

**Why it is included:** Catches the URL-fabrication failure mode (agent invents a plausible-looking source URL when the real source isn't available). Particularly useful in live mode because real search results have noisier coverage than hand-authored fixtures.

**How it is calculated:** Same GEval criteria as in L2 fixture mode. Reads `RETRIEVAL_CONTEXT` (which is the real `tool_output` in live mode).

---

<a id="l2live-judge-tokens"></a>
### Judge Tokens

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 (each of input / output / total) |
| Direction | lower is better |
| Toggle group | always-on (judge_tokens group) |
| Stored as | `eval_cases.judge_input_tokens` / `judge_output_tokens` / `judge_total_tokens`, with run-level averages on `eval_runs` |
| Computed by | `_TokenTrackingJudgeLLM` wrapper in `eval_engine.py` |
| Pass condition | (not gated) |

**What it evaluates:** Same as for L2 fixture mode: Tokens consumed by the judge LLM panel for one case. However, in live mode the panel is smaller (`delta_quality` and `significance_justification` are dropped), so per-case token cost is lower than in fixture mode.

**Why it is included:** Combined with `agent_*_tokens`, it gives full cost visibility for a live run.

**How it is calculated:** Same wrapper class as L2 fixture mode. One `_TokenTrackingJudgeLLM` per case, shared across the (smaller) live judge panel.

---

## Summary Metrics

These are run-level aggregates across cases surfaced at the top of each report and in the SQLite `eval_runs` row. They let you compare runs over time without drilling into per-case details.

<a id="pass-rate"></a>
### Pass Rate

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always (run-level summary) |
| Stored as | `eval_runs.pass_rate` |
| Computed by | `build_summary` in `eval_report_manager.py` |
| Pass condition | (not a gate; this is the result of all gates) |

**What it evaluates:** The fraction of cases in the run that ended with status `PASS` (i.e., every active gate succeeded). Cases that failed any gate end with status `REVIEW`.

**Why it is included:** The headline number for the question: Is this run good? Trend over time tells you whether the agent quality is regressing or improving.

**How it is calculated:** `pass_count / max(case_count, 1)`, rounded to `precision_for(_FMTS["pass_rate"])` decimals (4, since the registry format is `.2%` and percent display multiplies by 100). `pass_count` = number of results with `status == "PASS"`.

---

<a id="avg-judge-score"></a>
### Avg Judge Score

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always (run-level summary, only meaningful when judge group is enabled) |
| Stored as | `eval_runs.avg_judge_run_score` |
| Computed by | `build_summary` in `eval_report_manager.py` |
| Pass condition | (not gated) |

**What it evaluates:** Run-level mean of per-case `avg_judge_score`, where the per-case value is itself the mean of all judge metrics that produced a score for that case.

**Why it is included:** A single "judge quality" number per run, useful as a sanity check alongside pass rate. A high pass rate with a low judge score might mean gates are too lenient; a low pass rate with a high judge score might mean format/routing is breaking even though the content quality is fine.

**How it is calculated:** Per case: collect each non-None `.score` from `judge_metrics.values()` into a list, then `safe_mean(scores, precision_for(_FMTS["avg_judge_score"]))` — NULL when no judge produced a score. Run level: `safe_mean` of those per-case values, rounded to `precision_for(_FMTS["avg_judge_run_score"])` (NULL cases excluded). NULL for L1 (no judges run). Stored precision tracks the registry `fmt` for each key.

---

<a id="verdict-logic"></a>
## PASS/REVIEW Gates Verdict Logic

Each case ends with a `status` of either `PASS` or `REVIEW`. `PASS` means every active gate succeeded, while `REVIEW` means at least one gate failed and the case warrants a human look. The `failure_reasons` field on the result records every gate that fired (not just the first), so you can triage from the dashboard without re-running the case. The list of active gates depends on the variant, since some metrics are NULL or absent in some modes.

### Layer 1

L1 has its own inline gate logic in `evaluate_l1_case` (it doesn't share `_compute_status` with L2). Gates that can flip a case to REVIEW:

- `run produced errors` — any entry in the `errors` list.
- `agent hit recursion limit` — `recursion_limit_hit == True`.
- `report_length < min_report_length` — sub-agent response shorter than the `MIN_REPORT_LENGTH` floor configured in `eval_config.py`.
- `result_count < expected_min_results` — fewer URLs returned than the case requires.
- `excluded domain present in results` — any URL matched `EXCLUDED_DOMAINS`.
- `disallowed_keyword_hits = N` — at least one disallowed keyword found.
- `required_keyword_hit_rate < threshold` — too few required keywords matched.

L1 does NOT gate on `tool_call_count` (recorded as a diagnostic only), `domain_inclusion_present` (informational), or any judge / format / routing metrics (those don't apply at L1).

### Layer 2 fixture mode

L2 fixture mode uses `_compute_status` with a metric set tuned for the synthetic scenario. Active gates:

- `run produced errors` — typically empty in fixture mode (the agent doesn't run), but reserved for system-level issues.
- `agent hit recursion limit` — defensive gate; `recursion_limit_hit` is None in fixture mode (no agent runs), so the conditional never trips. Reserved so the gate code stays uniform across modes.
- `report_length < min_report_length` — also doubles as a fixture-authorship sanity check.
- `faithfulness < threshold` — judge gate.
- `answer_relevancy < threshold` — judge gate.
- `no_speculation < threshold` — judge gate (catches fabricated citation URLs that faithfulness only partially overlaps with).
- `delta_quality < threshold` — fixture-only judge gate. Tests the agent's core CI value prop (correctly attributing new-vs-old delta against the synthetic `history_context`). Skipped in live mode because the metric is None there.
- `significance_score_valid < 1.0` — score must be present and in [1, 10].
- `required_sections_present < 1.0` — all 5 mandatory sections must appear.
- `citation_presence < 1.0` — every Key Findings bullet must end in a real URL.
- `significance_score outside expected range for this scenario` — the synthetic-scenario delta gate.
- `disallowed_keyword_hits > 0`.
- `required_keyword_hit_rate < threshold`.

**Not gated in this mode:**

- `directive_compliance` — live-only (fixture's `tool_names` is hand-authored, not behavioral).
- `tool_call_count` — live-only for the same reason.
- `excluded domain present in report citations` — live-only (fixture report is hand-authored against possibly historical `EXCLUDED_DOMAINS` config; would test fixture authorship rather than agent behavior).
- `significance_justification` — judge metric is computed and recorded (score is stored on the case row and shown on the dashboard), but its value doesn't flip the verdict. `significance_score_valid` and `score_in_expected_range` already cover whether the score is well-formed and within the expected range, so this judge would mostly add noise on top.

### Layer 2 live mode

L2 live mode uses the same `_compute_status` path as fixture, with the following differences driven by which metrics are populated:

**Newly active in L2 live mode (real agent behavior to measure):**

- `agent hit recursion limit` — *can* actually fire (real run, real exception possible).
- `directive_compliance < 1.0` and `tool_call_count > expected_max_tool_calls` — both gates run against real tool-call traces.
- `excluded domain present in report citations` — symmetric with L1's excluded-domain gate, applied to URLs the supervisor cited in `final_report`. Catches cases where the agent surfaces an `EXCLUDED_DOMAINS` URL in Key Findings or Sources even when the search backend was clean.

**Also active in L2 live mode, same as L2 fixture mode:**

- Same judge gates (`faithfulness`, `answer_relevancy`, `no_speculation`).
- Same format gates (`significance_score_valid`, `required_sections_present`, `citation_presence`).
- Same keyword gates (`disallowed_keyword_hits == 0`, `required_keyword_hit_rate >= threshold`).
- Same `report_length` floor.
- Same `run produced errors` catch-all.

**NOT active in L2 live mode (no synthetic ground truth):**

- `significance_score outside expected range for this scenario` — `score_in_expected_range` is None in live mode (the synthetic `history_context` is stripped, so there's no expected-range yardstick).
- `delta_quality < threshold` — `delta_quality` is fixture-only (the metric requires `history_context` as CONTEXT, which is absent in live mode), so the score is None and the gate skips automatically.
