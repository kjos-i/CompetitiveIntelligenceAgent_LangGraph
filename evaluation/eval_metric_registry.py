"""Single source of truth for every metric in the CI agent evaluation harness.

Other modules import helpers from here instead of hardcoding metric names,
display labels, SQL columns, or CSV fieldnames. To rename, add, or remove a
metric, edit the METRICS list below. Every consumer derives its lists
from this registry automatically.

Metric groups
-------------
- judge           — LLM-judged DeepEval metrics, mix of built-in
                        (AnswerRelevancyMetric, FaithfulnessMetric) and
                        custom GEval (delta_quality,
                        significance_justification, no_speculation)
                        (toggle_group="judge")
- format          — Structural format compliance checks (toggle_group="format")
- routing         — Tool routing / stop-rule behavior (toggle_group="routing")
- delta           — Significance score extraction and range check
                        (no toggle; computed in fixture mode only,
                        NULL in live mode — see the comment in
                        evaluate_l2_case for why)
- keyword         — Required/disallowed keyword checks (always-on, composite)
- subagent        — Sub-agent isolation metrics. No toggle gates them,
                        but coverage varies by layer:
                        result_count and domain_inclusion_present are
                        L1-only (NULL on L2 rows);
                        domain_exclusion_respected is computed for L1
                        and L2 live (NULL in L2 fixture).
- diagnostic      — Signals that don't fit a policy group, no toggle
                        gates them. report_length is computed every run
                        (in fixture mode it measures the canned report).
                        recursion_limit_hit is NULL in L2 fixture mode
                        because no agent runs there — fixture mode reads
                        the canned report straight from JSON, so there's
                        no graph execution to exceed.
- judge_aggregate — Per-case aggregates across the judge panel
                        (avg_judge_score). Distinct from the run-level summary
                        which lives in eval_runs.metric_averages.
- tokens          — Agent LLM token consumption (input/output/total).
                        Captured for both layers — L2 from the supervisor
                        pipeline, L1 from the sub-agent in isolation. NULL
                        in fixture mode unless saved into the fixture.
- judge_tokens    — DeepEval judge LLM token consumption
                        (input/output/total). Captured for both L2 modes
                        (the only places judges run); NULL for L1.
- latency         — Wall-clock timing (NULL in fixture mode)

Toggle groups
-------------
toggle_group on a MetricDef controls whether the metric is computed at all
in a given run, driven by ENABLED_METRIC_GROUPS in eval_config.py.

  - "judge"   — DeepEval LLM-judged metrics
  - "format"  — Structural format checks
  - "routing" — Directive compliance and stop-rule

toggle_group=None means always-on regardless of ENABLED_METRIC_GROUPS.

Execution mode (fixture vs live) is NOT a toggle group. It is selected
per pass via RUN_VARIANTS in eval_config.py (each entry maps to a
specific layer + execution_mode combination). See the docstring there.

Adding a metric
---------------
1. Add a MetricDef entry to the METRICS list below.
2. Write the compute function in eval_metrics.py (deterministic) or add
   a new DeepEval metric to build_judge_metrics() in eval_engine.py.
3. Wire the metric into the engine:
   - Call your compute function inside the relevant case function(s) in
     eval_engine.py.
   - Add an entry to the returned result dict under the same key you
     used in the MetricDef.
   Pick which function(s) based on scope:
     - Layer 1 only  -> evaluate_l1_case()
     - Layer 2 only  -> evaluate_l2_case()
     - Both layers   -> both (mirror the key name and value type)
4. Everything else (SQLite schema, CSV, report averages, dashboard
   labels) updates automatically from the registry on next startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

MetricGroup = Literal[
    "judge", "format", "routing", "delta", "keyword", "subagent",
    "diagnostic", "judge_aggregate", "tokens", "judge_tokens", "latency",
]

ToggleGroup = Literal["judge", "format", "routing"]


@dataclass(frozen=True, slots=True)
class MetricDef:
    """Definition of a single evaluation metric."""

    key: str
    """Internal dict key used in result dicts and JSON reports."""

    label: str
    """Human-readable display label."""

    group: MetricGroup
    """Family classification. Lets consumers pull ordered metric subsets
    by family via keys_in_group — the dashboard uses this to assemble
    candidate metric lists for the score-distribution block, and the
    engine, report manager, and SQLite writer single out the 'judge'
    family for LLM-panel handling. Not used for CSV column ordering,
    that follows METRICS list order."""

    sql_column: str | None = None
    """Column name in the eval_cases table.  None for composites."""

    sql_type: str = "REAL"
    """SQL column type."""

    fmt: str = ".2f"
    """Python format spec for display."""

    summary_avg_key: str | None = None
    """Key in the eval_runs table for the run-level average."""

    summary_avg_fmt: str | None = None
    """Format spec for the run-level summary_avg_key value when it should
    differ from fmt. Used when the per-case value is an integer
    (fmt="d") but the run-level mean is a float, e.g. token counts use
    ".0f" (whole numbers without crashing on float means), and
    recursion_limit_hit uses ".2f" because its run-level mean is a
    hit-rate fraction. None falls back to fmt.
    """

    composite: bool = False
    """True for metrics whose value is a dict rather than a scalar."""

    composite_sql_columns: tuple[tuple[str, str], ...] | None = None
    """For composite metrics, the (column_name, sql_type) pairs that the
    composite expands into when generating the eval_cases schema and CSV
    columns. None for non-composites. Stored as a tuple of tuples (not a
    list) because MetricDef is frozen (mutable defaults aren't allowed).
    """

    composite_summary_avg_keys: tuple[tuple[str, str], ...] | None = None
    """For composite metrics, the (sub_key, avg_key) pairs that produce
    run-level averages. sub_key is the field inside the composite dict
    (e.g. "required_keyword_hit_rate"); avg_key is the column added
    to eval_runs and the key set in the summary dict
    (e.g. "avg_required_keyword_hit_rate"). Scalar metrics use
    summary_avg_key instead. None for non-composites or composites
    without run-level aggregation.
    """

    composite_labels: tuple[tuple[str, str], ...] | None = None
    """For composite metrics, the (sub_key, display_label) pairs used
    by metric_labels() to render dashboard headers, CSV column titles,
    and the Metrics Guide. The label is reused for the matching
    avg_key so per-case and run-level views share one display name.
    None falls back to the raw key (avoid for any sub-field that
    surfaces in user-facing UI).
    """

    toggle_group: ToggleGroup | None = None
    """Which group toggle in ENABLED_METRIC_GROUPS (eval_config.py)
    controls whether this metric is computed in a given run. When the
    group is absent from that set, the engine skips both the compute
    call and the verdict gate. None means always-on."""

    fixture_only: bool = False
    """True for metrics that only make sense in fixture mode (their
    criteria depend on synthetic ground truth that the live agent doesn't
    receive). The engine drops these from the judge panel in live runs.
    """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRICS: list[MetricDef] = [
    # ── LLM-judged (DeepEval) — toggle_group="judge" ─────────────────────────
    MetricDef(key="answer_relevancy",           label="Answer Relevancy",     group="judge",     sql_column="answer_relevancy",           toggle_group="judge"),
    MetricDef(key="faithfulness",               label="Faithfulness",         group="judge",     sql_column="faithfulness",               toggle_group="judge"),
    MetricDef(key="delta_quality",              label="Delta Quality",        group="judge",     sql_column="delta_quality",              toggle_group="judge", fixture_only=True),
    MetricDef(key="significance_justification", label="Significance Justification",  group="judge",     sql_column="significance_justification", toggle_group="judge", fixture_only=True),
    MetricDef(key="no_speculation",             label="No Speculation",       group="judge",     sql_column="no_speculation",             toggle_group="judge"),

    # ── Format compliance — toggle_group="format" ──────────────────────────
    MetricDef(key="required_sections_present",  label="Sections Present",     group="format",  sql_column="required_sections_present",  toggle_group="format"),
    MetricDef(key="citation_presence",          label="Citation Presence",    group="format",  sql_column="citation_presence",          toggle_group="format"),
    MetricDef(key="significance_score_valid",   label="Significance Valid",          group="format",  sql_column="significance_score_valid",   toggle_group="format", fmt=".0f", summary_avg_key="avg_significance_score_valid", summary_avg_fmt=".2f"),

    # ── Routing / behavior — toggle_group="routing" ──────────────────────
    MetricDef(key="tool_call_count",            label="Tool Call Count",      group="routing", sql_column="tool_call_count",       sql_type="INTEGER", toggle_group="routing", fmt="d", summary_avg_key="avg_tool_call_count", summary_avg_fmt=".2f"),
    MetricDef(key="directive_compliance",       label="Directive Compliance", group="routing", sql_column="directive_compliance",       toggle_group="routing", fmt=".0f", summary_avg_key="avg_directive_compliance", summary_avg_fmt=".2f"),

    # ── Delta scoring — no toggle; fixture-only, NULL in live ─────────────
    MetricDef(key="significance_score",         label="Significance Score",   group="delta",   sql_column="significance_score",    sql_type="INTEGER", fmt="d"),
    MetricDef(key="score_in_expected_range",    label="Significance in Range", group="delta",   sql_column="score_in_expected_range",    fmt=".0f"),

    # ── Keyword checks (composite, always-on) ──────────────────────────────
    MetricDef(
        key="keyword_checks",
        label="Keyword Checks",
        group="keyword",
        composite=True,
        composite_sql_columns=(
            ("required_keyword_hit_rate", "REAL"),
            ("disallowed_keyword_hits",   "INTEGER"),
        ),
        composite_summary_avg_keys=(
            ("required_keyword_hit_rate", "avg_required_keyword_hit_rate"),
            ("disallowed_keyword_hits",   "avg_disallowed_keyword_hits"),
        ),
        composite_labels=(
            ("required_keyword_hit_rate", "Keyword Hit Rate"),
            ("disallowed_keyword_hits",   "Disallowed Keywords"),
        ),
    ),

    # ── Diagnostics — no toggle ────────────────────────────────────────────
    # report_length is computed every run (in L2 fixture mode it measures
    # the canned report from JSON). recursion_limit_hit is NULL in L2
    # fixture mode because no agent runs there.
    MetricDef(key="report_length",              label="Report Length",        group="diagnostic", sql_column="report_length",       sql_type="INTEGER", fmt="d"),
    # Per-case is 0/1/NULL; the run-level summary_avg gives the hit rate
    # (mean of 0/1, with NULL fixture rows excluded). Useful for trend
    # charts answering "are recent runs looping more often?".
    MetricDef(key="recursion_limit_hit",        label="Recursion Limit Hit",  group="diagnostic", sql_column="recursion_limit_hit", sql_type="INTEGER", fmt="d", summary_avg_key="avg_recursion_limit_hit_rate", summary_avg_fmt=".2f"),

    # ── Sub-agent isolation — no toggle; coverage varies by layer ─────────
    # result_count and domain_inclusion_present are L1-only.
    # domain_exclusion_respected is populated for both L1 and L2 live
    # runs (the agent's final_report is checked symmetric with L1's backend
    # check); L2 fixture and L1-empty cases leave it None, which safe_mean
    # excludes from the run average.
    MetricDef(key="result_count",               label="Result URL Count",     group="subagent", sql_column="result_count",               sql_type="INTEGER", fmt="d", summary_avg_key="avg_result_count", summary_avg_fmt=".2f"),
    MetricDef(key="domain_exclusion_respected", label="Domain Exclusion",     group="subagent", sql_column="domain_exclusion_respected", summary_avg_key="avg_domain_exclusion_respected"),
    MetricDef(key="domain_inclusion_present",   label="Domain Inclusion",     group="subagent", sql_column="domain_inclusion_present",   summary_avg_key="avg_domain_inclusion_present"),

    # ── Per-case judge aggregates ──────────────────────────────────────────
    MetricDef(key="avg_judge_score",            label="Avg Judge Score",      group="judge_aggregate", sql_column="avg_judge_score"),

    # ── Agent token consumption — NULL in fixture mode unless captured ────
    MetricDef(key="agent_input_tokens",         label="Agent Input Tokens",   group="tokens",  sql_column="agent_input_tokens",  sql_type="INTEGER", fmt="d", summary_avg_key="avg_agent_input_tokens",  summary_avg_fmt=".0f"),
    MetricDef(key="agent_output_tokens",        label="Agent Output Tokens",  group="tokens",  sql_column="agent_output_tokens", sql_type="INTEGER", fmt="d", summary_avg_key="avg_agent_output_tokens", summary_avg_fmt=".0f"),
    MetricDef(key="agent_total_tokens",         label="Agent Total Tokens",   group="tokens",  sql_column="agent_total_tokens",  sql_type="INTEGER", fmt="d", summary_avg_key="avg_agent_total_tokens",  summary_avg_fmt=".0f"),

    # ── Judge token consumption — NULL for L1 (no judges run) ─────────────
    MetricDef(key="judge_input_tokens",         label="Judge Input Tokens",   group="judge_tokens", sql_column="judge_input_tokens",  sql_type="INTEGER", fmt="d", summary_avg_key="avg_judge_input_tokens",  summary_avg_fmt=".0f"),
    MetricDef(key="judge_output_tokens",        label="Judge Output Tokens",  group="judge_tokens", sql_column="judge_output_tokens", sql_type="INTEGER", fmt="d", summary_avg_key="avg_judge_output_tokens", summary_avg_fmt=".0f"),
    MetricDef(key="judge_total_tokens",         label="Judge Total Tokens",   group="judge_tokens", sql_column="judge_total_tokens",  sql_type="INTEGER", fmt="d", summary_avg_key="avg_judge_total_tokens",  summary_avg_fmt=".0f"),

    # ── Latency — NULL in fixture mode ─────────────────────────────────────
    MetricDef(key="latency_seconds",            label="Latency",              group="latency", sql_column="latency_seconds",      summary_avg_key="avg_latency_seconds",      fmt=".2f"),
]

# Fast lookup by key.
_BY_KEY: dict[str, MetricDef] = {metric.key: metric for metric in METRICS}

# ---------------------------------------------------------------------------
# Helper functions — every consumer imports from here
# ---------------------------------------------------------------------------

def judge_metric_keys() -> list[str]:
    """Ordered keys for the LLM-judged DeepEval metrics."""
    return [metric.key for metric in METRICS if metric.group == "judge"]


def keys_in_toggle_group(toggle_group: str) -> list[str]:
    """Return all metric keys assigned to the given toggle_group."""
    return [metric.key for metric in METRICS if metric.toggle_group == toggle_group]


def keys_in_group(group: str) -> list[str]:
    """Return the column-name keys belonging to a metric group.

    Composites are exploded into their concrete sub-column names (e.g.
    keyword_checks → ["required_keyword_hit_rate", "disallowed_keyword_hits"])
    so dashboards and reports can use the result directly as DataFrame
    column names without further unpacking. Non-composites return their
    own key.
    """
    out: list[str] = []
    for metric in METRICS:
        if metric.group != group:
            continue
        if metric.composite and metric.composite_sql_columns:
            out.extend(name for name, _ in metric.composite_sql_columns)
        else:
            out.append(metric.key)
    return out


def fixture_only_metric_keys() -> frozenset[str]:
    """Keys of metrics that should be skipped in live execution mode.

    Currently populated from the fixture_only flag on each MetricDef.
    The engine drops these from the judge panel when execution_mode
    is "live"; everything else (deterministic metrics, format/routing
    gates) is unaffected.
    """
    return frozenset(metric.key for metric in METRICS if metric.fixture_only)


@cache
def metric_labels() -> dict[str, str]:
    """key -> display label mapping, including SQL and avg-key aliases.

    Composite metrics expose their sub-fields via composite_labels;
    the same label is reused for the matching entry in
    composite_summary_avg_keys so per-case and run-level views of the
    same field share one display name.
    """
    labels: dict[str, str] = {metric.key: metric.label for metric in METRICS}
    for metric in METRICS:
        if metric.sql_column and metric.sql_column != metric.key:
            labels[metric.sql_column] = metric.label
        if metric.summary_avg_key:
            labels[metric.summary_avg_key] = metric.label
        if metric.composite_labels:
            sub_label_map = dict(metric.composite_labels)
            for sub_key, label in metric.composite_labels:
                labels[sub_key] = label
            # Map each composite avg_key to the same label as its source
            # sub_key, so dashboard headers for "avg_required_keyword_hit_rate"
            # render as "Keyword Hit Rate" rather than the raw key.
            if metric.composite_summary_avg_keys:
                for sub_key, avg_key in metric.composite_summary_avg_keys:
                    if sub_key in sub_label_map:
                        labels[avg_key] = sub_label_map[sub_key]
    labels["case_id"] = "Case ID"
    labels["layer"] = "Layer"
    labels["category"] = "Category"
    labels["status"] = "Status"
    labels["avg_judge_run_score"] = "Avg Judge Score (Run)"
    labels["pass_rate"] = "Pass Rate"
    return labels


@cache
def metric_fmts() -> dict[str, str]:
    """key -> format spec mapping for display formatting.

    Mirrors metric_labels(): propagates the per-case fmt to the
    matching summary_avg_key (using summary_avg_fmt when set,
    else falling back to fmt), to each composite_summary_avg_keys
    avg_key, and to each composite_sql_columns sub-field. Run-level
    keys not tied to any per-case metric (pass_rate,
    avg_judge_run_score) are hand-added at the bottom.
    """
    fmts: dict[str, str] = {metric.key: metric.fmt for metric in METRICS}
    for metric in METRICS:
        if metric.summary_avg_key:
            fmts[metric.summary_avg_key] = metric.summary_avg_fmt or metric.fmt
        if metric.composite_sql_columns:
            for col_name, _ in metric.composite_sql_columns:
                fmts[col_name] = metric.fmt
        if metric.composite_summary_avg_keys:
            for _sub_key, avg_key in metric.composite_summary_avg_keys:
                fmts[avg_key] = metric.fmt
    fmts["pass_rate"]           = ".1%"
    fmts["avg_judge_run_score"] = ".2f"
    return fmts


def case_sql_columns() -> list[tuple[str, str]]:
    """Ordered (column_name, sql_type) pairs for metric columns in eval_cases.

    Composites are exploded into their concrete sub-columns via the
    composite_sql_columns field on each MetricDef — adding a new
    composite metric just ships its expansion alongside its definition,
    no special-casing here.
    """
    columns: list[tuple[str, str]] = []
    for metric in METRICS:
        if metric.composite:
            if metric.composite_sql_columns:
                columns.extend(metric.composite_sql_columns)
        elif metric.sql_column:
            columns.append((metric.sql_column, metric.sql_type))
    return columns


def run_sql_columns() -> list[tuple[str, str]]:
    """Ordered (column_name, sql_type) pairs for average columns in eval_runs.

    Includes scalar averages (summary_avg_key) and composite averages
    (composite_summary_avg_keys) so adding a composite avg in the
    registry auto-extends the eval_runs schema on next startup via the
    _try_add_column migration loop.
    """
    columns: list[tuple[str, str]] = [
        (metric.summary_avg_key, "REAL") for metric in METRICS if metric.summary_avg_key
    ]
    for metric in METRICS:
        if metric.composite_summary_avg_keys:
            for _sub_key, avg_key in metric.composite_summary_avg_keys:
                columns.append((avg_key, "REAL"))
    return columns


def summary_avg_pairs() -> list[tuple[str, str]]:
    """(summary_avg_key, source_key) pairs for building scalar run summaries.

    Composite averages live in composite_summary_avg_specs() because
    their source value is nested inside a composite dict on the per-case
    result (e.g. result["keyword_checks"]["required_keyword_hit_rate"]).
    """
    return [(metric.summary_avg_key, metric.key) for metric in METRICS if metric.summary_avg_key]


def composite_summary_avg_specs() -> list[tuple[str, str, str]]:
    """(composite_key, sub_key, avg_key) triples for composite averages.

    composite_key is the top-level field on each per-case result that
    holds the composite dict (e.g. "keyword_checks"); sub_key is
    the field inside that dict to aggregate; avg_key is the run-level
    name written to the summary dict and the eval_runs row.
    """
    specs: list[tuple[str, str, str]] = []
    for metric in METRICS:
        if metric.composite_summary_avg_keys:
            for sub_key, avg_key in metric.composite_summary_avg_keys:
                specs.append((metric.key, sub_key, avg_key))
    return specs


def csv_fieldnames() -> list[str]:
    """Ordered column names for the per-case CSV export.

    session_id is repeated on every row of a single run's CSV — it's
    the link back to eval_runs.session_id in SQLite and to the matching
    JSON report's top-level session_id field, so a CSV shared out of
    context is self-describing. scenario_type is L2-only and stays
    empty on L1 rows; notes is freeform per-case commentary;
    failure_reasons is the joined list of gates that fired
    (semicolon-separated in _build_csv_row) so a REVIEW verdict is
    self-explanatory in the same row.
    """
    preamble = ["session_id", "id", "layer", "category", "scenario_type", "notes", "status"]
    metric_cols = [col for col, _ in case_sql_columns()]
    trailing = ["error_count", "failure_reasons"]
    return preamble + metric_cols + trailing
