"""Configuration for the CI agent evaluation harness.

Edit the values below to control how evaluation runs behave.

Run variants
------------
RUN_VARIANTS is a list of independent evaluation passes to run in a
single invocation. Each entry produces its own JSON + CSV report and its
own row in the SQLite ledger. All entries in one invocation share the
same session_id so they can be grouped after the fact in the
dashboard.

Valid entries:

  "l1" — Layer 1 sub-agent isolation. Always live (its
    purpose is to exercise the real Brave/Tavily API integration).
    No fixture mode exists for this layer.

  "l2_fixture" — Layer 2 end-to-end agent run against hand-authored
    fixtures in eval_cases_layer2.json. Each fixture is paired with
    a synthetic history_context and expected_answer_points to
    form a self-consistent scenario, so the full metric panel,
    including delta judgments and the score-range gate, applies. The
    agent's LLM still runs (to test supervisor synthesis), but no
    Brave/Tavily API calls are made. Deterministic, free, continuious
    integration-safe. Latency stored as NULL.

  "l2_live" — Layer 2 end-to-end agent run against real Brave/Tavily
    APIs. Non-deterministic, costs money. The synthetic
    history_context is stripped (the agent runs as a cold-start
    research task) and metrics that depend on synthetic ground truth,
    delta_quality and score_in_expected_range, are not
    computed (stored as NULL). Functional metrics (faithfulness,
    no_speculation, answer_relevancy, significance_justification,
    format gates, routing gates, keyword checks) all run normally.

Mix any subset. ["l1", "l2_fixture", "l2_live"] runs all three back
to back. ["l2_fixture"] runs just the cheap deterministic check.

ENABLED_METRIC_GROUPS
---------------------
Controls which toggleable metric groups are computed in layer 2 runs:
  "judge"   — DeepEval LLM-judged metrics (5 metrics, costs LLM calls)
  "format"  — Structural format checks (deterministic, always cheap)
  "routing" — Directive compliance and stop-rule checks (deterministic)

Always-on regardless of this setting:
  delta metrics, keyword_checks, avg_judge_score, latency (NULL in fixture mode)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# --- Paths ---
LAYER1_DATASET_PATH = Path(__file__).resolve().parent / "eval_cases_layer1.json"
LAYER2_DATASET_PATH = Path(__file__).resolve().parent / "eval_cases_layer2.json"
OUTPUT_DIR          = Path(__file__).resolve().parent / "evaluation_results"

# --- Judge settings ---
JUDGE_MODEL     = "gpt-4o-mini"
JUDGE_THRESHOLD = 0.5

# --- Verdict gate thresholds ---
# Do NOT change these between runs once you have started collecting trend data.
# The active thresholds are persisted with each run, and the dashboard warns on drift.
#
# Binary-by-design gates (not configurable):
#   - significance_score_valid must be 1.0 (format requirement)
#   - directive_compliance must be 1.0
#   - disallowed_keyword_hits must be 0
REQUIRED_KEYWORD_THRESHOLD = 0.5

# Reports shorter than this are flagged as REVIEW. Catches outputs that pass
# the section-regex gate but are otherwise truncated/empty (e.g. the supervisor
# emitted a stub like "## Executive Summary\n\nSIGNIFICANCE_SCORE: 1"). Tune
# upward if your reports are reliably longer. 200 is a defensive floor.
MIN_REPORT_LENGTH = 200

# --- Run variants ---
# Each entry is a fully independent evaluation pass with its own JSON + CSV
# and its own ledger row. All entries in one invocation share a session_id
# so they can be grouped in the dashboard. See module docstring for the
# semantics of each variant.
RunVariant = Literal["l1", "l2_fixture", "l2_live"]


@dataclass(frozen=True)
class VariantSpec:
    """Static metadata for a single RunVariant.

    Centralised so the runner's dispatch loop, the dashboard's variant
    label/order logic, and the _variant_label helper all agree on
    layer / execution_mode / display string for a given variant key.
    Adding a fourth variant means adding one entry here and everything
    downstream picks it up.
    """

    layer: int
    execution_mode: str
    display: str


# Order matters — the dashboard uses dict insertion order as the canonical
# display order for variants in the session dropdown and case counts.
VARIANT_SPECS: dict[str, VariantSpec] = {
    "l1":         VariantSpec(layer=1, execution_mode="live",    display="L1"),
    "l2_fixture": VariantSpec(layer=2, execution_mode="fixture", display="L2 fixture"),
    "l2_live":    VariantSpec(layer=2, execution_mode="live",    display="L2 live"),
}

RUN_VARIANTS: list[RunVariant] = ["l1", "l2_fixture", "l2_live"]

_invalid_variants = set(RUN_VARIANTS) - set(VARIANT_SPECS.keys())
if _invalid_variants:
    raise ValueError(
        f"Invalid entries in RUN_VARIANTS: {sorted(_invalid_variants)}. "
        f"Valid values are: {sorted(VARIANT_SPECS.keys())}"
    )

# --- Run settings ---
CONCURRENCY = 1
MAX_CASES: int | None = 10   # Set to an int to limit cases evaluated

# Cap on simultaneous in-flight judge HTTP calls. With CONCURRENCY=2 cases each
# firing 5 GEval metrics in parallel, up to 10 calls could otherwise be in flight
# at once. 2 leaves headroom under typical OpenAI per-minute limits without
# serialising work that's safely within budget. Bump CONCURRENCY without bumping
# this value to throttle judge calls intentionally.
JUDGE_CONCURRENCY = 1

# --- Metric groups (Layer 2 only) ---
ENABLED_METRIC_GROUPS: set[str] = {"judge", "format", "routing"}

_VALID_METRIC_GROUPS: set[str] = {"judge", "format", "routing"}
_invalid_groups = ENABLED_METRIC_GROUPS - _VALID_METRIC_GROUPS
if _invalid_groups:
    raise ValueError(
        f"Invalid entries in ENABLED_METRIC_GROUPS: {sorted(_invalid_groups)}. "
        f"Valid values are: {sorted(_VALID_METRIC_GROUPS)}"
    )

# --- Full metric groups required for SQLite persistence ---
# Only runs where all three groups are enabled are written to the ledger,
# so trend charts compare like-with-like.
FULL_METRIC_GROUPS: frozenset[str] = frozenset({"judge", "format", "routing"})
