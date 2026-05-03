# Project Structure

A short, top-down map of the Competitive Intelligence Agent project. For
detail on individual files, see [README.md](README.md).

```
CompetitiveIntelligenceAgent_LangGraph/
│
├── README.md                          ← Project overview, setup, run instructions
├── info_project.md                    ← This file — folder structure at a glance
├── info_considerations.md             ← Strengths, trade-offs, and room for improvement
├── requirements.txt                   ← Pinned Python dependencies
├── system_prompt_agent.txt            ← Supervisor system prompt
├── watchlist.json                     ← Competitors monitored in Auto mode
│
├── launch_agent.py                    ← CLI entry — Manual or Auto mode
├── launch_schedule_runner.py          ← APScheduler entry — recurring automation
│
├── agent.py                           ← LangGraph supervisor + sub-agents + run_agent()
├── agent_modes.py                     ← run_manual_chat() / run_automated_lookout()
├── config.py                          ← All runtime tuning (model, search, schedule)
│
├── memory_sqlite3.py                  ← SQLite persistence layer (intel_ledger table)
├── memory_ledger_db.py                ← CLI viewer for ledger entries
├── pydantic_models.py                 ← Company, SentimentResult schemas
├── utils.py                           ← Logger, graph PNG export, watchlist loader
│
├── dashboard.py                       ← Streamlit dashboard for the intel ledger
│
├── ⟨runtime artifacts, generated on first run⟩
│   ├── agent_memory.db                ← SQLite intel ledger
│   ├── agent.log                      ← Runtime log
│   └── agent_graph.png                ← Optional graph diagram (when DRAW=True)
│
├── tests/                             ← pytest suite for the parent project
│   ├── conftest.py
│   ├── test_pydantic_models.py
│   ├── test_memory_sqlite3.py
│   └── test_utils.py
│
└── evaluation/                        ← Self-contained evaluation harness
    │
    ├── README.md                      ← Harness overview, variants, run instructions
    ├── info_metrics.md                ← Per-metric reference + verdict-gate logic
    │
    ├── eval_runner.py                 ← CLI entry — dispatches each RUN_VARIANT
    ├── eval_config.py                 ← Harness tuning (variants, thresholds, concurrency)
    │
    ├── eval_pydantic_models.py        ← L1EvalCase, L2EvalCase Pydantic schemas
    ├── eval_metric_registry.py        ← Single source of truth — every MetricDef
    ├── eval_metrics.py                ← Pure compute_* functions (deterministic metrics)
    ├── eval_utils.py                  ← Loaders, text extraction, math helpers
    │
    ├── eval_engine.py                 ← Layer 1 + Layer 2 case runners, judge panel
    ├── eval_report_manager.py         ← Per-run summary → JSON / CSV artifacts
    ├── eval_sqlite.py                 ← SQLite ledger (eval_runs, eval_cases tables)
    ├── eval_dashboard.py              ← Streamlit dashboard for harness results
    │
    ├── eval_cases_layer1.json         ← Layer 1 sub-agent isolation cases
    ├── eval_cases_layer2.json         ← Layer 2 full-pipeline cases (incl. fixtures)
    │
    ├── ⟨runtime artifacts, generated on first run⟩
    │   ├── eval_ledger.db             ← SQLite ledger of every evaluation run
    │   └── evaluation_results/        ← Timestamped JSON + CSV reports per run
    │
    └── tests/                         ← pytest suite for the evaluation harness
        ├── conftest.py
        ├── test_eval_utils.py
        ├── test_eval_metrics.py
        └── test_eval_metric_registry.py
```

## Layout at a glance

The parent project and the `evaluation/` subproject are **structurally
symmetric**. Each has its own runner, config, models, persistence
layer, dashboard, and pytest suite.

| Concern | Parent | `evaluation/` |
|---|---|---|
| Entry point | `launch_agent.py`, `launch_schedule_runner.py` | `eval_runner.py` |
| Tuning | `config.py` | `eval_config.py` |
| Schemas | `pydantic_models.py` | `eval_pydantic_models.py` |
| Persistence | `memory_sqlite3.py` | `eval_sqlite.py` |
| Dashboard | `dashboard.py` | `eval_dashboard.py` |
| Tests | `tests/` | `evaluation/tests/` |

Runtime artifacts (`*.db`, `*.log`, generated `*_results/`) are
created on first run and are safe to delete or `.gitignore`.
