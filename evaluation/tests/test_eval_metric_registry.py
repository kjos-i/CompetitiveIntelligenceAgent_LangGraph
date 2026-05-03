"""Consistency tests for the metric registry.

These tests guard the invariants that other modules silently rely on
(every per-case key has a format spec, every composite has matching
labels and SQL columns, etc.). A regression here would otherwise
manifest as missing dashboard cards or KeyErrors in CSV / SQLite
serialisation.
"""

from __future__ import annotations

from eval_metric_registry import (
    METRICS,
    case_sql_columns,
    composite_summary_avg_specs,
    csv_fieldnames,
    judge_metric_keys,
    keys_in_group,
    keys_in_toggle_group,
    metric_fmts,
    metric_labels,
    run_sql_columns,
    summary_avg_pairs,
)


def test_all_metric_keys_unique():
    keys = [m.key for m in METRICS]
    assert len(keys) == len(set(keys)), "Duplicate MetricDef.key"


def test_every_metric_has_label_and_fmt():
    labels = metric_labels()
    fmts = metric_fmts()
    for metric in METRICS:
        assert metric.key in labels, f"{metric.key} missing from metric_labels()"
        assert metric.key in fmts, f"{metric.key} missing from metric_fmts()"


def test_summary_avg_keys_are_unique():
    avg_keys = [m.summary_avg_key for m in METRICS if m.summary_avg_key]
    assert len(avg_keys) == len(set(avg_keys)), "Duplicate summary_avg_key"


def test_summary_avg_keys_have_fmt_and_label():
    fmts = metric_fmts()
    labels = metric_labels()
    for metric in METRICS:
        if metric.summary_avg_key:
            assert metric.summary_avg_key in fmts
            assert metric.summary_avg_key in labels


def test_composite_metrics_have_columns_and_labels():
    for metric in METRICS:
        if metric.composite:
            assert metric.composite_sql_columns, (
                f"{metric.key} is composite but has no composite_sql_columns"
            )
            # Labels should cover every sub-column the composite expands into.
            sub_columns = {name for name, _ in metric.composite_sql_columns}
            label_subs = (
                {sub for sub, _ in metric.composite_labels}
                if metric.composite_labels
                else set()
            )
            assert sub_columns == label_subs, (
                f"{metric.key} composite_labels do not cover sub-columns "
                f"(missing: {sub_columns - label_subs})"
            )


def test_composite_summary_specs_align_with_columns():
    for metric in METRICS:
        if not metric.composite_summary_avg_keys:
            continue
        sub_keys_in_columns = {name for name, _ in (metric.composite_sql_columns or ())}
        for sub_key, avg_key in metric.composite_summary_avg_keys:
            assert sub_key in sub_keys_in_columns, (
                f"{metric.key}: composite avg sub_key {sub_key!r} "
                f"is not a declared composite_sql_column"
            )
            assert avg_key.startswith("avg_"), avg_key


def test_judge_metric_keys_all_in_judge_group():
    for key in judge_metric_keys():
        metric = next(m for m in METRICS if m.key == key)
        assert metric.toggle_group == "judge"


def test_keys_in_toggle_group_match_toggle_field():
    for group in ("judge", "format", "routing"):
        expected = [m.key for m in METRICS if m.toggle_group == group]
        assert keys_in_toggle_group(group) == expected


def test_keys_in_group_explodes_composites():
    keyword_keys = keys_in_group("keyword")
    # Composite keyword_checks should expand into its two sub-columns.
    assert "required_keyword_hit_rate" in keyword_keys
    assert "disallowed_keyword_hits" in keyword_keys
    assert "keyword_checks" not in keyword_keys


def test_csv_fieldnames_are_unique():
    fields = csv_fieldnames()
    assert len(fields) == len(set(fields)), "Duplicate CSV fieldnames"


def test_csv_fieldnames_include_every_metric_column():
    fields = set(csv_fieldnames())
    for col, _ in case_sql_columns():
        assert col in fields, f"case column {col!r} missing from csv_fieldnames()"


def test_run_sql_columns_match_summary_pairs():
    declared = {col for col, _ in run_sql_columns()}
    pair_keys = {avg_key for avg_key, _ in summary_avg_pairs()}
    composite_avgs = {avg for _, _, avg in composite_summary_avg_specs()}
    # Every average key surfaces as a column on eval_runs.
    assert pair_keys.issubset(declared)
    assert composite_avgs.issubset(declared)


def test_pass_rate_format_is_percent():
    fmts = metric_fmts()
    assert fmts["pass_rate"].endswith("%"), (
        "pass_rate is conceptually a fraction — its display format "
        "should be a percent so 0.85 renders as 85.00%."
    )
