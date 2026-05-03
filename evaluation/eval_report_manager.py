"""Report generation and persistence for the CI agent evaluation harness."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_metric_registry import (
    METRICS,
    composite_summary_avg_specs,
    csv_fieldnames,
    judge_metric_keys,
    keys_in_toggle_group,
    metric_fmts,
    metric_labels,
    summary_avg_pairs,
)
from eval_utils import precision_for, safe_mean

# Snapshot of the registry's format table at import time (metric_fmts
# is @cache-decorated so a fresh import reflects the current registry).
_FMTS = metric_fmts()


class ReportManager:
    """Handles evaluation summaries and persisted reports."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def build_summary(
        self,
        results: list[dict[str, Any]],
        judge_model: str,
        enabled_groups: set[str] | None = None,
        execution_mode: str = "fixture",
    ) -> dict[str, Any]:
        """Aggregate per-case results into a top-level summary dict."""

        # Judge metric averages: Scores live in result["judge_metrics"][key]
        metric_keys = sorted({
            metric_key
            for result_dict in results
            for metric_key in result_dict.get("judge_metrics", {}).keys()
        })
        result_averages: dict[str, Any] = {}

        for metric_key in metric_keys:
            scores = [
                result_dict["judge_metrics"][metric_key]["score"]
                for result_dict in results
                if result_dict.get("judge_metrics", {}).get(metric_key, {}).get("score") is not None
            ]
            result_averages[metric_key] = safe_mean(scores, precision_for(_FMTS.get(metric_key, ".2f")))

        # Format / routing / delta scores live as top-level keys on each
        # per-case result rather than inside judge_metrics. Skip any
        # key that already has a summary_avg_key, those are handled
        # by the summary_avg_pairs loop below and writing them here
        # too would duplicate the entry under both the raw key and the
        # avg_* key.
        keys_with_summary_avg = {m.key for m in METRICS if m.summary_avg_key}
        for result_key in keys_in_toggle_group("format") + keys_in_toggle_group("routing"):
            if result_key in keys_with_summary_avg:
                continue
            scores = [
                result_dict[result_key] for result_dict in results
                if result_dict.get(result_key) is not None
            ]
            if scores:
                result_averages[result_key] = safe_mean(scores, precision_for(_FMTS.get(result_key, ".2f")))

        pass_count = sum(1 for result_dict in results if result_dict["status"] == "PASS")
        case_scores = [result_dict["avg_judge_score"] for result_dict in results if result_dict.get("avg_judge_score") is not None]

        summary: dict[str, Any] = {
            "case_count":     len(results),
            "pass_count":     pass_count,
            "pass_rate":      round(pass_count / max(len(results), 1), precision_for(_FMTS["pass_rate"])),
            # Computed here rather than in the summary_avg_pairs loop because
            # it's a top-level run-wide score and the source values live on
            # the per-case avg_judge_score key, not in any registry pair.
            "avg_judge_run_score": safe_mean(case_scores, precision_for(_FMTS["avg_judge_run_score"])) if case_scores else None,
        }

        for avg_key, src_key in summary_avg_pairs():
            scores = [
                result_dict[src_key]
                for result_dict in results
                if result_dict.get(src_key) is not None
            ]
            summary[avg_key] = safe_mean(scores, precision_for(_FMTS.get(avg_key, ".2f"))) if scores else None

        # Composite averages: source values are nested one level deep
        # (e.g. result["keyword_checks"]["required_keyword_hit_rate"])
        # so they need a different access path than the scalar pairs.
        for composite_key, sub_key, avg_key in composite_summary_avg_specs():
            scores = [
                result_dict.get(composite_key, {}).get(sub_key)
                for result_dict in results
                if result_dict.get(composite_key, {}).get(sub_key) is not None
            ]
            summary[avg_key] = safe_mean(scores, precision_for(_FMTS.get(avg_key, ".2f"))) if scores else None

        summary["judge_model"]     = judge_model
        summary["execution_mode"]  = execution_mode
        summary["enabled_groups"]  = sorted(enabled_groups) if enabled_groups is not None else None
        summary["metric_averages"] = result_averages

        return summary

    def save_report(
        self,
        report: dict[str, Any],
        label: str = "layer2",
    ) -> tuple[Path, Path]:
        """Write timestamped JSON and CSV reports to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"eval_report_{label}_{timestamp}.json"
        csv_path  = self.output_dir / f"eval_summary_{label}_{timestamp}.csv"

        # ensure_ascii=False preserves non-ASCII characters as literal
        # glyphs instead of escape sequences so the JSON stays
        # human-readable when reports include them.
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        fieldnames = csv_fieldnames()
        llm_keys   = set(judge_metric_keys())
        session_id = report.get("session_id")

        # extrasaction="ignore" lets _build_csv_row return extra
        # keys (e.g. notes) without raising — only fields listed in
        # fieldnames are written, in that order.
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for item in report.get("results", []):
                writer.writerow(_build_csv_row(item, fieldnames, llm_keys, session_id))

        return json_path, csv_path

    @staticmethod
    def print_summary(
        summary: dict[str, Any],
        json_path: Path,
        csv_path: Path,
        label: str = "Layer 2",
    ) -> None:
        """Print a formatted summary of the evaluation run to stdout."""
        not_evaluated = "Not evaluated"

        def _fmt(value: Any, suffix: str = "") -> str:
            return not_evaluated if value is None else f"{value}{suffix}"

        print("\n" + "=" * 80)
        print(f"EVAL SUMMARY — COMPETITIVE INTELLIGENCE AGENT ({label.upper()})")
        print("=" * 80)
        print(f"Cases evaluated      : {summary['case_count']}")
        print(f"Pass rate            : {summary['pass_count']}/{summary['case_count']} ({summary['pass_rate']:.1%})")
        print(f"Avg judge score      : {_fmt(summary.get('avg_judge_run_score'))}")

        mode = summary.get("execution_mode")
        if mode:
            print(f"Execution mode       : {mode}")

        enabled = summary.get("enabled_groups")
        if enabled is not None:
            print(f"Enabled metric groups: {', '.join(enabled) if enabled else '(none)'}")

        labels = metric_labels()
        for avg_key, _ in summary_avg_pairs():
            if avg_key in summary and summary[avg_key] is not None:
                label_str = labels.get(avg_key, avg_key)
                print(f"{label_str:21s}: {_fmt(summary[avg_key])}")

        print(f"Judge model          : {summary['judge_model']}")
        print("Metric averages      :")
        metric_averages = summary.get("metric_averages") or {}
        if metric_averages:
            for name, score in metric_averages.items():
                print(f"  - {name}: {_fmt(score)}")
        else:
            print(f"  {not_evaluated}")

        print(f"JSON report          : {json_path}")
        print(f"CSV summary          : {csv_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_csv_row(
    item: dict[str, Any],
    fieldnames: list[str],
    llm_keys: set[str],
    session_id: str | None,
) -> dict[str, Any]:
    """Flatten one per-case result dict into a single CSV row.

    Driving the row from the registry-derived fieldnames (instead of
    a hardcoded literal) means adding a metric in
    eval_metric_registry flows automatically into the CSV.

    Five field shapes are special-cased because they don't live as
    scalars on the case dict: LLM judge scores nest under
    judge_metrics[name]['score']; keyword checks nest under
    keyword_checks; error_count is derived from the errors
    list length; failure_reasons is a list flattened to one
    semicolon-separated cell so REVIEW verdicts stay self-explanatory
    in spreadsheets; and session_id is a run-level value injected
    from the report (repeated on every row so a CSV shared out of
    context can be linked back to the JSON / SQLite row). Everything
    else is a top-level scalar pulled via item.get(field).
    """
    metrics = item.get("judge_metrics", {})
    keywords = item.get("keyword_checks", {})
    row: dict[str, Any] = {}
    for field in fieldnames:
        if field == "session_id":
            row[field] = session_id
        elif field in llm_keys:
            row[field] = metrics.get(field, {}).get("score")
        elif field in ("required_keyword_hit_rate", "disallowed_keyword_hits"):
            row[field] = keywords.get(field)
        elif field == "error_count":
            row[field] = len(item.get("errors", []))
        elif field == "failure_reasons":
            row[field] = "; ".join(item.get("failure_reasons", []))
        else:
            row[field] = item.get(field)
    return row
