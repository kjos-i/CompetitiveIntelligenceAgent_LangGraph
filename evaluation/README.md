# CI Agent — Evaluation Harness

Evaluation harness for the Competitive Intelligence (CI) agent. Runs the agent against curated test cases, scores its output with a mix of deterministic checks and LLM-judge metrics (DeepEval), persists every run to a SQLite ledger, and surfaces the results in a Streamlit dashboard for trend analysis.

The harness has **three execution variants**, configured via `RUN_VARIANTS` in `eval_config.py`:

- **`l1`** — Layer 1 sub-agent isolation. Calls `brave_agent` or `tavily_agent` directly against `eval_cases_layer1.json`. Always live (its purpose is to verify the real Brave / Tavily API integration). Tests structural properties of the search-tool response (URL counts, domain rules, keyword presence).
- **`l2_fixture`** — Layer 2 end-to-end agent run replay against hand-authored fixtures in `eval_cases_layer2.json`. The agent's LLM does *not* run. The panel of metrics (including LLM judges) is replayed against the stored report. Cheap, deterministic, and CI-safe (safe to run in continuous integration). Ideal for iterating on judge prompts and gate logic.
- **`l2_live`** — Layer 2 end-to-end agent run against the real Brave / Tavily APIs. Non-deterministic, costs money. The synthetic `history_context` is stripped (cold-start) and metrics that depend on synthetic ground truth are stored as NULL.

A single invocation can run any subset of these variants. All variants in one invocation share a `session_id` so the dashboard can group them.

All scripts in this folder are Learning/Demo status.

See [info_metrics.md](info_metrics.md) for the per-metric reference (what each score means, and which gates flip a case from PASS to REVIEW per variant).

The harness's Streamlit dashboard can be used to visualize evaluation results in the SQLite ledger at any time:

```bash
python -m streamlit run evaluation/eval_dashboard.py --server.port 8502
```

## Why the metric registry pattern

Every metric in the harness is declared once as a `MetricDef` in [`eval_metric_registry.py`](eval_metric_registry.py). The registry is the single source of truth for:

- Display label and format specification (used by the dashboard's metric cards, per-case tables, CSV, and printed summary. `NumberColumn` formats flow from `metric_fmts()` via the dashboard's `column_format` helper, so it is never hardcoded).
- SQL column name and type (used to generate `eval_cases` and `eval_runs` schemas).
- Toggle group (`judge` / `format` / `routing`). Controls whether the metric is computed in a given run.
- Composite expansion (e.g. `keyword_checks` → `required_keyword_hit_rate` + `disallowed_keyword_hits`).
- Run-level summary aliases. Every per-case metric whose run-level mean has different display precision than the per-case value declares both `summary_avg_key` (e.g. `avg_tool_call_count`) and `summary_avg_fmt` (e.g. `".2f"` for the float mean of an integer count).

Adding a new metric means adding one entry to the `METRICS` list and writing the compute function (deterministic) or DeepEval criteria (LLM-judged). The SQLite schema, CSV columns, dashboard cards, and summary averages all derive from the registry automatically.

## Folder Contents

### Core Python Scripts

| File | Purpose | Status |
|---|---|---|
| [`eval_runner.py`](eval_runner.py) | CLI entry point — iterates `RUN_VARIANTS` and dispatches each to the matching layer / mode handler. All variants in one invocation share a `session_id`. | Learning/Demo |
| [`eval_config.py`](eval_config.py) | Central control panel — paths, judge model, thresholds, `RUN_VARIANTS`, concurrency, `ENABLED_METRIC_GROUPS`, `MAX_CASES`. | Learning/Demo |
| [`eval_pydantic_models.py`](eval_pydantic_models.py) | Pydantic models for the case datasets: `L1EvalCase` (sub-agent isolation), `L2EvalCase` (end-to-end agent run). | Learning/Demo |
| [`eval_metric_registry.py`](eval_metric_registry.py) | **Single source of truth** for every metric. Helper functions (`metric_fmts`, `metric_labels`, `csv_fieldnames`, `case_sql_columns`, `run_sql_columns`, `summary_avg_pairs`, …) are derived from this registry and consumed by every other module. | Learning/Demo |
| [`eval_metrics.py`](eval_metrics.py) | Pure `compute_*` functions for every deterministic metric (format compliance, citation presence, score extraction, keyword checks, domain rules, URL extraction). No LLM calls. | Learning/Demo |
| [`eval_utils.py`](eval_utils.py) | Generic helpers — dataset loading (`load_l1_cases`, `load_l2_cases`), text normalization, LangGraph message parsing (`content_to_text`, `extract_messages`, `extract_all_tool_outputs`, `join_tool_outputs`), DeepEval context builders (`build_expected_output`, `build_gold_context`, `build_retrieval_context`), and math (`safe_mean`, `precision_for`). | Learning/Demo |
| [`eval_engine.py`](eval_engine.py) | Runs both layers. `evaluate_l1_case` calls a sub-agent directly; `evaluate_l2_case` runs the end-to-end agent (supervisor + sub-agent, fixture or live), assembles the DeepEval test case, runs the judge panel, computes the verdict via `_compute_status`, and emits a per-case result dict. Token tracking via `_TokenTrackingJudgeLLM`. | Learning/Demo |
| [`eval_report_manager.py`](eval_report_manager.py) | Aggregates per-case results into a run summary, writes timestamped JSON + CSV reports to `evaluation_results/`, and prints a formatted summary to stdout. | Learning/Demo |
| [`eval_sqlite.py`](eval_sqlite.py) | SQLite ledger (`eval_ledger.db`) with two tables: `eval_runs` (one row per evaluation run) and `eval_cases` (one row per case per run). Schema generated from the registry; adding a new metric extends the schema automatically on next startup. | Learning/Demo |
| [`eval_dashboard.py`](eval_dashboard.py) | Streamlit dashboard for browsing runs, comparing variants in a session, drilling into per-case results, and watching trends over time. Reads directly from `eval_ledger.db`. | Learning/Demo |

### Documentation

| File | Purpose |
|---|---|
| [`info_metrics.md`](info_metrics.md) | Per-metric reference: what each metric evaluates, why it's included, how it's calculated, range, direction, pass condition. Also includes the per-variant verdict logic (which gates flip a case to REVIEW). Also rendered in the dashboard's "Metrics Guide" tab. |

### Datasets

| File | Purpose |
|---|---|
| [`eval_cases_layer1.json`](eval_cases_layer1.json) | Layer 1 cases — sub-agent isolation tests. Each case targets one search engine (`brave` / `tavily`) and includes `expected_min_results`, `required_keywords`, `disallowed_keywords`. |
| [`eval_cases_layer2.json`](eval_cases_layer2.json) | Layer 2 cases — end-to-end agent run tests. Each case includes a `query`, `history_context` (PRIOR STATE), `expected_answer_points` (gold context), `expected_significance_range`, optional hand-authored `fixture`, plus the same keyword and routing affordances as L1. |

### Test Suite

| File | Purpose | Test count |
|---|---|---|
| [`tests/conftest.py`](tests/conftest.py) | Adds the evaluation folder to `sys.path` so test modules can `import eval_*` without packaging the harness. | — |
| [`tests/test_eval_utils.py`](tests/test_eval_utils.py) | Unit tests for `normalize_text`, `precision_for`, `safe_mean`, `extract_message_text`, `extract_all_tool_outputs`, `content_to_text`, `extract_messages`, `join_tool_outputs`. | 45 |
| [`tests/test_eval_metrics.py`](tests/test_eval_metrics.py) | Unit tests for every deterministic `compute_*` function (format gates, score extraction, keyword checks, domain rules, URL extraction). | 53 |
| [`tests/test_eval_metric_registry.py`](tests/test_eval_metric_registry.py) | Consistency invariants — every metric has label and fmt, composite columns and labels align, judge keys all in judge group, summary avg keys are unique, CSV fieldnames cover every metric column, etc. | 13 |

Run the full suite from the evaluation folder:

```bash
python -m pytest tests/ -q
```

### Runtime Artifacts

| Path | Type | Purpose |
|---|---|---|
| `eval_ledger.db` | SQLite database | Persisted ledger of every evaluation run — created automatically on first run by `eval_sqlite.EvalLedger`. |
| `evaluation_results/` | Directory | Timestamped JSON + CSV reports, one pair per variant per session (e.g. `eval_report_layer2_20260428_142530.json`). |
| `__pycache__/`, `.pytest_cache/`, `.deepeval/` | Cache directories | Generated by Python / pytest / DeepEval. Safe to delete. |

## Architecture Overview

```
eval_runner.py
    └── for each variant in RUN_VARIANTS:
            EvaluationEngine(execution_mode=...)
                ├── evaluate_l1_cases()  ← live sub-agent, structural metrics only
                └── evaluate_l2_cases()  ← supervisor pipeline + judge panel
                        ├── fixture mode: read final_report from case.fixture
                        └── live mode:    astream_events through agent.py supervisor
                                ↓
                    DeepEval judge panel (registry-driven, fixture vs live)
                                ↓
                    _compute_status() → PASS / REVIEW + failure_reasons
                                ↓
            ReportManager.build_summary() → JSON + CSV in evaluation_results/
                                ↓
            EvalLedger.save_run()         → eval_ledger.db
                                ↓
                    eval_dashboard.py (Streamlit)
```

**Flow per variant:**

1. `eval_runner.py` resolves the variant from [`VARIANT_SPECS`](eval_config.py) (layer + execution_mode + display label) and constructs an `EvaluationEngine`.
2. The engine loads the matching dataset via `load_l1_cases` / `load_l2_cases`, optionally truncating to `MAX_CASES`.
3. Cases run concurrently (bounded by `CONCURRENCY`); LLM judge calls are further bounded by `JUDGE_CONCURRENCY`.
4. Each per-case result includes a verdict (`status: PASS|REVIEW`), a `failure_reasons` list of every gate that fired, errors collected during the run, and every metric value.
5. `ReportManager` aggregates per-case results into a summary, prints it to stdout, and writes both a timestamped JSON and CSV to `evaluation_results/`.
6. `EvalLedger.save_run` writes to SQLite, but **only if all three metric groups (`judge`, `format`, `routing`) were enabled**, so trend charts compare like-with-like. Partial runs still produce JSON/CSV artifacts.
7. All variants in one invocation share a `session_id` so the dashboard can group them as one session in the dropdown.

## Verdict Logic (PASS / REVIEW)

Each case ends with `status: PASS` (every active gate succeeded) or `status: REVIEW` (at least one gate failed, warrants a human look). The `failure_reasons` field records every gate that fired, not just the first, so the dashboard can surface the full list without re-running the case.

The set of active gates differs per variant. L2 fixture mode skips routing gates because `tool_names` is hand-authored, L2 live mode skips delta gates because there is no synthetic `history_context`, and L1 has its own inline gate logic in `evaluate_l1_case` that does not share the L2 path. See [info_metrics.md → PASS/REVIEW Gates Verdict Logic](info_metrics.md#verdict-logic) for the full per-variant breakdown.

## Configuration

All tuning lives in [`eval_config.py`](eval_config.py). See that file for the current values.

| Setting | Description |
|---|---|
| `RUN_VARIANTS` | Which variants to run in one invocation. Each runs as a separate pass with its own JSON/CSV/ledger row, all sharing one `session_id`. Mix any subset of `l1`, `l2_fixture`, `l2_live`. |
| `ENABLED_METRIC_GROUPS` | Layer 2 toggle groups (`judge`, `format`, `routing`). Only runs where all three are enabled are persisted to SQLite (so trend charts stay apples-to-apples). |
| `JUDGE_MODEL` | OpenAI model used for every DeepEval judge metric. |
| `JUDGE_THRESHOLD` | Minimum score for a judge metric to count as passed. |
| `REQUIRED_KEYWORD_THRESHOLD` | Minimum required-keyword hit rate (per case). |
| `MIN_REPORT_LENGTH` | Reports shorter than this trip the verdict to REVIEW. Catches truncated outputs that pass the section regex. |
| `CONCURRENCY` | Max cases evaluated in parallel. |
| `JUDGE_CONCURRENCY` | Max in-flight judge HTTP calls across all cases. Throttle independently of `CONCURRENCY`. |
| `MAX_CASES` | Cap on cases per variant; set to `None` to run the full dataset. |

### Run variants and where to add cases

| Variant | Dataset | When to use |
|---|---|---|
| `l1` | `eval_cases_layer1.json` | Verifying the search-tool integration (Brave / Tavily) — does the API return enough URLs, respect domain filters, surface required keywords? Always live. |
| `l2_fixture` | `eval_cases_layer2.json` (cases with a `fixture` block) | Iterating on judge prompts, format gates, or scoring rules without burning agent or search budget. Free, deterministic, CI-safe (safe to run in continuous integration). |
| `l2_live` | `eval_cases_layer2.json` | End-to-end regression check against the real supervisor + sub-agents. Non-deterministic; reserved for periodic / pre-release runs. |

Cases without a `fixture` block are skipped automatically in `l2_fixture` mode.

### Verdict-gate thresholds and trend stability

The `eval_runs.gate_thresholds` JSON column persists `JUDGE_THRESHOLD`, `REQUIRED_KEYWORD_THRESHOLD`, and `MIN_REPORT_LENGTH` with each run. The dashboard warns when these change between sessions because trend comparisons across threshold changes are not apples-to-apples. Be aware that **if you do change thresholds between runs once you have started collecting trend data, the runs can no longer be directly compared.**

## Adding a new metric

1. Add a `MetricDef` entry to `METRICS` in [`eval_metric_registry.py`](eval_metric_registry.py), specifying `key`, `label`, `group`, and (as needed) `sql_column`, `sql_type`, `fmt`, `summary_avg_key`, `summary_avg_fmt`, `toggle_group`, `composite_*` fields, `fixture_only`. When the per-case value is an integer or binary 0/1 (`fmt="d"` or `fmt=".0f"`), but the run-level mean is a fraction, set `summary_avg_fmt=".2f"` so the mean is stored and displayed with the precision it deserves (see `tool_call_count`, `result_count`, `significance_score_valid`, and `directive_compliance` for examples).
2. Write the compute function:
   - Deterministic → add `compute_<name>(...)` to [`eval_metrics.py`](eval_metrics.py).
   - LLM-judged → add a `DeepEval(...)` entry to `EvaluationEngine.build_judge_metrics` in [`eval_engine.py`](eval_engine.py).
3. Surface the value on each per-case result dict in `evaluate_l1_case` / `evaluate_l2_case`.
4. (Optional) Add a card to the dashboard if the new metric has a dedicated card position, otherwise it shows up automatically in metric averages and the per-case tables.
5. Save the `eval_ledger.db` aside under a new name (or wipe it if the current data is disposable) if your change renames or removes an existing column (`_try_add_column` is add-only by design).

The CSV fieldnames, SQLite schema, run summary averages, and the consistency invariants tested in `tests/test_eval_metric_registry.py` all derive from the registry, so they update automatically.

## Run Instructions

### Run the harness

From the evaluation folder (Python virtualenv with `requirements.txt` installed):

```bash
python eval_runner.py
```

The runner prints a per-variant summary to stdout, writes JSON + CSV artifacts to `evaluation_results/`, and persists each run to `eval_ledger.db`. L1 runs are always persisted. L2 runs are persisted only when all three metric groups (`judge`, `format`, `routing`) are enabled. Partial-group L2 runs still produce JSON + CSV but no ledger row.

### Run the test suite

```bash
python -m pytest tests/ -q
```

The unit test suite covers the deterministic metric functions, the math / text helpers, and the registry's consistency invariants. See the per-file counts in the [Test Suite](#test-suite) table above.

### Open the Streamlit dashboard

From the parent project folder:

```bash
python -m streamlit run evaluation/eval_dashboard.py --server.port 8502
```

The dashboard polls `eval_ledger.db` at the interval set by `POLL_SECONDS` in [`eval_dashboard.py`](eval_dashboard.py) and refreshes when new rows arrive. Tabs:

- **Run Summary** — Averages per variant + per-case PASS/REVIEW table + answer details. Per-case tables and the answer-detail right rails are variant-aware: Each L2 variant hides the metrics that are structurally NULL for that variant (e.g. fixture mode hides `tool_call_count` / `directive_compliance` / `latency_seconds`, and live mode hides the delta-comparison fields like `delta_quality`, `significance_justification`, `score_in_expected_range`, and `significance_score`). The data is still computed-as-NULL and persisted to the ledger, this functions purely as a display filter.
- **Deep Analysis** — Score-distribution histograms, per-case latency and token bars.
- **Historical Trends** — Pass rate, judge score, latency, and token charts over time.
- **Metrics Guide** — Renders [`info_metrics.md`](info_metrics.md) inline.

## Requirements

The harness shares its dependency stack with the parent agent project. In addition to the parent's `requirements.txt`, the harness uses:

- `deepeval` — LLM-judge metric framework (Answer Relevancy, Faithfulness, GEval).
- `pytest` — Unit test runner.

```bash
pip install -r ../requirements.txt
pip install deepeval pytest
```

## Environment Variables

For `l2_live` (and `l1`, since L1 always runs live), the agent's API keys must be available:

```env
OPENAI_API_KEY=...
BRAVE_SEARCH_API_KEY=...
TAVILY_API_KEY=...
```

`eval_runner.py` calls `load_dotenv()` before importing the engine, so the parent project's `.env` is picked up automatically.

`l2_fixture` only needs `OPENAI_API_KEY` (for the judge LLM), the search APIs are not called.

## Schema Policy

The SQLite ledger auto-migration is **add-only**. A new metric in the registry extends the schema on next startup, but renames and removals are not migrated. When a registry change renames or removes a column, the right fix is to save the `eval_ledger.db` with a new name before the next run (or wipe it if the current data is disposable). The harness itself does not support backward-compatibility shims.

## Suggested Learning Path

1. Read [`info_metrics.md`](info_metrics.md) end-to-end to understand what each metric measures and how the verdict logic works per variant.
2. Read [`eval_metric_registry.py`](eval_metric_registry.py) to see how every metric is declared in one place.
3. Read [`eval_config.py`](eval_config.py) to see what's tunable.
4. Run `python -m pytest tests/ -q` to verify the harness is healthy in your environment.
5. Run `python eval_runner.py` with `RUN_VARIANTS = ["l2_fixture"]` and `MAX_CASES = 2` for a fast, cheap end-to-end test.
6. Open the Streamlit dashboard to see the resulting session.
7. Read [`eval_engine.py`](eval_engine.py) to understand how cases are run and how the verdict gate works.
8. Read [`eval_report_manager.py`](eval_report_manager.py) and [`eval_sqlite.py`](eval_sqlite.py) to see how registry-derived schema flows through to artifacts.
