"""Entry point for the CI agent evaluation harness.

Usage
-----
Run from the evaluation folder:

    python eval_runner.py

Configuration is controlled entirely via eval_config.py:
    - RUN_VARIANTS     — list of any subset of {"l1", "l2_fixture", "l2_live"};
                         each entry is an independent pass with its own
                         JSON+CSV+ledger row, all linked by one session_id
    - MAX_CASES        — int to limit cases, or None to run all
    - ENABLED_METRIC_GROUPS — which toggle groups to compute
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

# Load env vars BEFORE importing eval_engine — agent.py instantiates LLM
# clients at import time and reads OPENAI_API_KEY / search-API keys from
# os.environ, so deferring this to main() would race the import.
load_dotenv()

# Resolve imports relative to the evaluation folder.
_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from eval_config import (
    CONCURRENCY,
    ENABLED_METRIC_GROUPS,
    JUDGE_CONCURRENCY,
    JUDGE_MODEL,
    JUDGE_THRESHOLD,
    LAYER1_DATASET_PATH,
    LAYER2_DATASET_PATH,
    MAX_CASES,
    MIN_REPORT_LENGTH,
    OUTPUT_DIR,
    REQUIRED_KEYWORD_THRESHOLD,
    RUN_VARIANTS,
    VARIANT_SPECS,
)
from eval_engine import EvaluationEngine
from eval_report_manager import ReportManager
from eval_sqlite import EvalLedger
from eval_utils import load_l1_cases, load_l2_cases


async def run_layer1(
        engine: EvaluationEngine, 
        report_manager: ReportManager, 
        ledger: EvalLedger, 
        max_cases: int | None, 
        session_id: str | None = None
) -> None:
    """Run layer 1 sub-agent isolation cases."""

    cases = load_l1_cases(LAYER1_DATASET_PATH)
    if max_cases:
        cases = cases[:max_cases]

    print(f"\n[Layer 1] Running {len(cases)} sub-agent isolation case(s) (always live)...")

    results = await engine.evaluate_l1_cases(cases, concurrency=CONCURRENCY)
    summary = report_manager.build_summary(
        results,
        judge_model=JUDGE_MODEL,
        enabled_groups=set(),          # No LLM judge in Layer 1.
        execution_mode="live",
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "session_id":   session_id,
        "layer":        1,
        "judge_model":  JUDGE_MODEL,
        "threshold":    JUDGE_THRESHOLD,
        "dataset_path": str(LAYER1_DATASET_PATH),
        "summary":      summary,
        "results":      results,
    }

    json_path, csv_path = report_manager.save_report(report, label="layer1")
    report_manager.print_summary(summary, json_path, csv_path, label="Layer 1")

    run_id = ledger.save_l1_run(report, session_id=session_id)
    print(f"\nLedger: L1 run saved as run_id={run_id}.")


async def run_layer2(
    engine: EvaluationEngine,
    report_manager: ReportManager,
    ledger: EvalLedger,
    max_cases: int | None,
    execution_mode: str,
    session_id: str | None = None,
) -> None:
    """Run layer 2 end-to-end agent run cases."""

    cases = load_l2_cases(LAYER2_DATASET_PATH)
    if max_cases:
        cases = cases[:max_cases]

    # In fixture mode, skip cases with no hand-authored fixture.
    if execution_mode == "fixture":
        runnable = [case for case in cases if case.fixture is not None]
        skipped  = len(cases) - len(runnable)
        if skipped:
            print(
                f"\n[Layer 2] Skipping {skipped} case(s) with no fixture. "
                "Add a 'fixture' block to the case in eval_cases_layer2.json "
                "or include 'l2_live' in RUN_VARIANTS."
            )
        cases = runnable

    if not cases:
        print("\n[Layer 2] No cases to run.")
        return

    print(
        f"\n[Layer 2] Running {len(cases)} case(s) "
        f"(mode={execution_mode}, judge={JUDGE_MODEL}, "
        f"groups={sorted(ENABLED_METRIC_GROUPS)})..."
    )

    results = await engine.evaluate_l2_cases(cases, concurrency=CONCURRENCY)
    summary = report_manager.build_summary(
        results,
        judge_model=JUDGE_MODEL,
        enabled_groups=ENABLED_METRIC_GROUPS,
        execution_mode=execution_mode,
    )

    report = {
        "generated_at":   datetime.now().isoformat(),
        "session_id":     session_id,
        "layer":          2,
        "judge_model":    JUDGE_MODEL,
        "threshold":      JUDGE_THRESHOLD,
        "execution_mode": execution_mode,
        "dataset_path":   str(LAYER2_DATASET_PATH),
        "gate_thresholds": engine.gate_thresholds(),
        "summary":        summary,
        "results":        results,
    }

    json_path, csv_path = report_manager.save_report(report, label="layer2")
    report_manager.print_summary(summary, json_path, csv_path, label="Layer 2")

    # Persist to SQLite ledger when all metric groups were enabled.
    run_id = ledger.save_run(report, execution_mode=execution_mode, session_id=session_id)
    if run_id:
        print(f"\nLedger: run saved as run_id={run_id}.")
    else:
        print(
            "\nLedger: run NOT saved (not all metric groups were enabled or "
            "not all groups in FULL_METRIC_GROUPS). JSON/CSV artifacts are still written."
        )


async def main() -> None:
    """Iterate RUN_VARIANTS and dispatch each to its layer/mode handler.

    A fresh EvaluationEngine is built per variant because
    execution_mode is set at engine construction time and gates the
    judge panel and metric computation. All variants in this invocation
    share one session_id so the dashboard can group their separately
    persisted ledger rows back together.
    """
    report_manager = ReportManager(OUTPUT_DIR)
    ledger         = EvalLedger()
    session_id     = uuid4().hex

    def make_engine(execution_mode: str) -> EvaluationEngine:
        return EvaluationEngine(
            judge_model=JUDGE_MODEL,
            threshold=JUDGE_THRESHOLD,
            required_keyword_threshold=REQUIRED_KEYWORD_THRESHOLD,
            min_report_length=MIN_REPORT_LENGTH,
            enabled_groups=ENABLED_METRIC_GROUPS,
            execution_mode=execution_mode,
            judge_concurrency=JUDGE_CONCURRENCY,
        )

    if not RUN_VARIANTS:
        print("\nRUN_VARIANTS is empty in eval_config.py. Nothing to do.")
        return

    print(f"\nSession {session_id} — running variants: {list(RUN_VARIANTS)}")

    for variant in RUN_VARIANTS:
        spec = VARIANT_SPECS[variant]
        engine = make_engine(spec.execution_mode)
        if spec.layer == 1:
            await run_layer1(engine, report_manager, ledger, MAX_CASES, session_id)
        else:
            await run_layer2(engine, report_manager, ledger, MAX_CASES, spec.execution_mode, session_id)


if __name__ == "__main__":
    asyncio.run(main())
