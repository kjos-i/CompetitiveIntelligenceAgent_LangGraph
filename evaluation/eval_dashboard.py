"""Streamlit evaluation dashboard for the CI agent evaluation harness.

Launch from the evaluation folder (or the project root):
    streamlit run evaluation/eval_dashboard.py

No custom CSS or HTML is used — every widget is a native Streamlit component.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import altair as alt  # noqa: F401  (kept for future Altair charts)
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Path setup — allow importing from the evaluation folder directly
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from eval_metric_registry import (
    judge_metric_keys,
    keys_in_group,
    keys_in_toggle_group,
    metric_fmts,
    metric_labels,
)
from eval_config import VARIANT_SPECS
from eval_engine import EvaluationEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLL_SECONDS  = 10
DB_PATH       = _EVAL_DIR / "eval_ledger.db"

JUDGE_KEYS      = judge_metric_keys()
LABELS        = metric_labels()

# Pass-through of the registry's format table. The single override
# below covers agent_total_tokens, whose registry fmt is "d"
# but which the dashboard renders as a float (it's summed from two mean
# token counts). .0f works on both int and float so the card never
# falls through fmt_val's str() fallback.
FMTS: dict[str, str] = {
    **metric_fmts(),
    "agent_total_tokens": ".0f",
}

# All metric-key lists below are derived from the registry so adding,
# renaming, or removing a metric in eval_metric_registry.py automatically
# flows through to every dashboard widget that uses these constants.
FORMAT_KEYS   = keys_in_toggle_group("format")
ROUTING_KEYS  = keys_in_toggle_group("routing")
DELTA_KEYS    = keys_in_group("delta")
SUBAGENT_KEYS = keys_in_group("subagent")
KEYWORD_KEYS  = keys_in_group("keyword")
LATENCY_KEYS  = keys_in_group("latency")
TOKEN_KEYS        = keys_in_group("tokens")
JUDGE_TOKEN_KEYS  = keys_in_group("judge_tokens")
DIAGNOSTIC_KEYS   = keys_in_group("diagnostic")
JUDGE_AGGREGATE_KEYS = keys_in_group("judge_aggregate")

NOT_EVALUATED = "—"

# Canonical display order for the variants emitted by the runner. Pulled
# from VARIANT_SPECS' insertion order so the dashboard never disagrees with
# the runner about which variants exist or how they should be labelled.
_CANONICAL_VARIANTS: tuple[str, ...] = tuple(spec.display for spec in VARIANT_SPECS.values())


def _variant_label(layer: object, mode: object) -> str:
    """Map a (layer, execution_mode) pair to the runner's variant label."""
    for spec in VARIANT_SPECS.values():
        if spec.layer == layer and spec.execution_mode == mode:
            return spec.display
    return str(mode) if mode else NOT_EVALUATED

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Eval Dashboard — CI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shrink the metric value font so 7 cards fit on one row, and allow
# long labels to wrap instead of being clipped. Every nested wrapper
# inside stMetricLabel is overridden because the inner elements
# default to nowrap/ellipsis. min-height reserves space for a
# second line so single- and two-line cards align vertically.
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] label * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricLabel"] {
        min-height: 2.4em !important;
        max-width: 12ch !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=POLL_SECONDS)
def _get_latest_timestamp() -> str | None:
    """Cheaply poll for the newest run — used to decide when to bust caches."""
    if not DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT MAX(run_timestamp) FROM eval_runs").fetchone()
            return row[0] if row else None
    except sqlite3.Error:
        return None


@st.cache_data(show_spinner=False)
def load_runs() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM eval_runs WHERE layer = 2 OR layer IS NULL ORDER BY run_timestamp DESC", conn
        )
    if not df.empty:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_l1_runs() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM eval_runs WHERE layer = 1 ORDER BY run_timestamp ASC", conn
        )
    if not df.empty:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_l1_cases() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM eval_cases WHERE layer = 1 ORDER BY run_timestamp ASC, case_id", conn
        )
    if not df.empty:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_cases() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM eval_cases ORDER BY run_timestamp DESC, case_id", conn
        )
    if not df.empty:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    return df


def build_sessions(l1_runs: pd.DataFrame, l2_runs: pd.DataFrame) -> pd.DataFrame:
    """One row per session_id, aggregating L1 + L2 ledger rows.

    A session = one invocation of eval_runner.py. With multiple variants
    in RUN_VARIANTS, a session writes one ledger row per variant — all
    sharing a session_id — so the dashboard treats the session as the
    primary unit and the dropdown lists sessions, not raw rows.
    """
    frames = [df for df in (l1_runs, l2_runs) if not df.empty]
    if not frames:
        return pd.DataFrame(columns=["session_id", "run_timestamp", "variants",
                                      "judge_model", "run_count"])
    all_runs = pd.concat(frames, ignore_index=True)
    all_runs["variant"] = all_runs.apply(
        lambda r: _variant_label(r.get("layer"), r.get("execution_mode")), axis=1
    )

    rows: list[dict[str, Any]] = []
    for sid, grp in all_runs.groupby("session_id", dropna=False, sort=False):
        present = set(grp["variant"])
        variants = [v for v in _CANONICAL_VARIANTS if v in present]
        variants += sorted(v for v in present if v not in _CANONICAL_VARIANTS)
        # Sessions written before session_id was added land here as NaN —
        # synthesize a per-row pseudo-id so they each show as their own
        # entry in the dropdown rather than collapsing into one giant group.
        if pd.isna(sid):
            sid = f"_legacy_{int(grp['run_id'].iloc[0])}"
        judge_models = grp["judge_model"].dropna()
        rows.append({
            "session_id":    sid,
            "run_timestamp": grp["run_timestamp"].max(),
            "variants":      variants,
            "judge_model":   judge_models.iloc[0] if not judge_models.empty else None,
            "run_count":     len(grp),
        })
    return (
        pd.DataFrame(rows)
          .sort_values("run_timestamp", ascending=False)
          .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_val(value: object, fmt: str = ".2f", suffix: str = "") -> str:
    """Return a formatted string, or NOT_EVALUATED for None/NaN."""
    if value is None:
        return NOT_EVALUATED
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return NOT_EVALUATED
    except (TypeError, ValueError):
        pass
    try:
        return f"{value:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return str(value)


def fmt_for(key: str, value: object, suffix: str = "") -> str:
    """Format value using the registry-driven format spec for key.

    Looks up FMTS[key] (the registry's per-case fmts plus dashboard
    summary-key overrides), falling back to .2f if the key isn't
    registered. Use this instead of hardcoding format literals in the
    metric cards so renaming or re-formatting in eval_metric_registry.py
    propagates automatically.
    """
    return fmt_val(value, FMTS.get(key, ".2f"), suffix)


def column_format(key: str, *, suffix: str = "") -> str:
    """Convert the registry's Python format spec for key into the
    printf-style spec st.column_config.NumberColumn expects.

    Mirrors fmt_for but returns the format string itself (e.g.
    "%.2f", "%d", "%.2fs") for use in column configs, so
    table columns inherit the registry's per-metric formatting without
    hardcoding %.2f / %d literals at the call site.
    """
    return f"%{FMTS.get(key, '.2f')}{suffix}"


def delta_str(
    current: object,
    previous: object,
    fmt: str = ".2f",
) -> str | None:
    """Return a signed delta string, or None when values are unavailable."""
    try:
        c = float(current)  # type: ignore[arg-type]
        p = float(previous)  # type: ignore[arg-type]
        if pd.isna(c) or pd.isna(p):
            return None
        return f"{c - p:+{fmt}}"
    except (TypeError, ValueError):
        return None


def _prev(prev_row: pd.Series | None, col: str) -> object:
    """Safely read a column from the previous run row (a pandas Series)."""
    if prev_row is None:
        return None
    try:
        val = prev_row[col]
        return None if pd.isna(val) else val  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        return None


def _parse_json_col(raw: object) -> dict[str, Any]:
    """Parse a JSON-serialised dict column, returning {} on failure."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


# ---------------------------------------------------------------------------
# Conditional cell colouring (per-case results tables)
# ---------------------------------------------------------------------------

# CSS used by the score and latency band tables. Pulled out as constants
# so the gradient is described in one place.
_GREEN_CSS  = "color: #4ade80; font-weight: 600"
_ORANGE_CSS = "color: #fb923c; font-weight: 600"
_RED_CSS    = "color: #f87171; font-weight: 600"

# Default judge threshold used when a session's gate_thresholds JSON is
# missing or malformed. Matches the eval_config default so historical rows
# without a recorded threshold render the same gradient they did before.
_DEFAULT_JUDGE_THRESHOLD = 0.5


def _score_bands(judge_threshold: float) -> list[tuple[float, str]]:
    """Build best→worst color bands keyed off the active judge threshold.

    Green starts at the midpoint between the threshold and 1.0 so the band
    scales with whatever threshold the run was scored against — a 0.6 score
    is "borderline" when the threshold is 0.5 but "failing" when the
    threshold is 0.7. Without this, the dashboard's colors stay anchored to
    a hardcoded 0.5 cutoff and silently mislead anyone reading a
    higher-threshold run.
    """
    return [
        ((judge_threshold + 1.0) / 2, _GREEN_CSS),
        (judge_threshold,             _ORANGE_CSS),
        (0.0,                         _RED_CSS),
    ]


_LATENCY_BANDS: list[tuple[float, str]] = [
    (5,     _GREEN_CSS),
    (10,    _ORANGE_CSS),
    (99999, _RED_CSS),
]


def _color_by_thresholds(
    val: object,
    bands: list[tuple[float, str]],
    *,
    higher_is_better: bool = True,
) -> str:
    """Return a CSS color string based on threshold bands."""
    if not isinstance(val, (int, float)) or pd.isna(val):
        return "color: #6b7280"
    for cutoff, css in bands:
        if (higher_is_better and val >= cutoff) or (not higher_is_better and val <= cutoff):
            return css
    return bands[-1][1] if bands else ""


# ---------------------------------------------------------------------------
# Plotly chart builders (used by the Deep Analysis tab)
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT_DEFAULTS: dict[str, object] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=16),
    margin=dict(l=40, r=20, t=30, b=40),
)


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_radar_chart(df: pd.DataFrame, metric_cols: list[str], color: str = "#636EFA") -> go.Figure:
    """Radar/polar chart of mean metric scores (each metric expected in 0–1)."""
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return go.Figure()

    means = df[available].apply(pd.to_numeric, errors="coerce").mean()
    labels = [LABELS.get(c, c) for c in available]
    values = means.tolist()

    # Close the polygon
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        fillcolor=_hex_to_rgba(color, alpha=0.12),
        line=dict(color=color, width=2),
        marker=dict(size=5),
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        polar=dict(
            radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0], gridcolor="#e9ecef"),
            angularaxis=dict(gridcolor="#e9ecef"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=320,
        margin=dict(l=110, r=110, t=70, b=70),
    )
    return fig


def build_latency_bar(df: pd.DataFrame, group_col: str | None = None) -> go.Figure | None:
    """Bar of per-case latency (NULLs are dropped).

    When group_col is provided, cases are sorted by that column and the
    bars are colored per group so they read as one chart with two clusters
    (e.g. all L1 cases, then all L2 live cases).
    """
    if "latency_seconds" not in df.columns:
        return None

    has_group = bool(group_col) and group_col in df.columns
    cols      = ["case_id", "latency_seconds"] + ([group_col] if has_group else [])
    data      = df[cols].copy().dropna(subset=["latency_seconds"])
    if data.empty:
        return None

    if has_group:
        data["_x"] = data[group_col].astype(str) + " \u00b7 " + data["case_id"].astype(str)
        order = (
            data.drop_duplicates("_x")
                .sort_values([group_col, "case_id"])["_x"]
                .tolist()
        )
        fig = px.bar(
            data, x="_x", y="latency_seconds", color=group_col,
            color_discrete_map={"L1": "#636EFA", "L2 live": "#00CC96"},
            category_orders={"_x": order},
        )
        fig.update_layout(legend_title="Layer")
    else:
        fig = px.bar(
            data, x="case_id", y="latency_seconds",
            color_discrete_sequence=["#636EFA"],
        )
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="",
        yaxis_title="Seconds",
        yaxis=dict(gridcolor="#f0f0f0"),
        height=370,
        bargap=0.1,
    )
    return fig


def build_token_bar(df: pd.DataFrame, group_col: str | None = None) -> go.Figure | None:
    """Stacked bar of input vs output tokens per case (NULLs are dropped).

    When group_col is provided, cases are sorted by that column and a
    visual gap is inserted between groups so they read as one chart with
    two clusters (e.g. all L1 cases, then all L2 live cases) rather than
    two side-by-side facets.
    """
    token_cols = [c for c in ("agent_input_tokens", "agent_output_tokens") if c in df.columns]
    if not token_cols:
        return None

    has_group = bool(group_col) and group_col in df.columns
    id_vars   = ["case_id"] + ([group_col] if has_group else [])
    data      = df[id_vars + token_cols].copy()
    melted    = data.melt(id_vars=id_vars, var_name="component", value_name="tokens")
    melted    = melted.dropna(subset=["tokens"])
    if melted.empty:
        return None
    melted["component"] = melted["component"].map({
        "agent_input_tokens":  "Input",
        "agent_output_tokens": "Output",
    }).fillna(melted["component"])

    # When grouping, build a composite x label "<group> · <case_id>" so case
    # IDs that appear in both groups (none today, but possible) stay distinct,
    # and order the categories so all of one group sits before the other.
    if has_group:
        melted["_x"] = melted[group_col].astype(str) + " · " + melted["case_id"].astype(str)
        order = (
            melted.drop_duplicates("_x")
                  .sort_values([group_col, "case_id"])["_x"]
                  .tolist()
        )
        x_field = "_x"
    else:
        order   = None
        x_field = "case_id"

    fig = px.bar(
        melted, x=x_field, y="tokens", color="component", barmode="stack",
        color_discrete_map={"Input": "#636EFA", "Output": "#00CC96"},
        category_orders={x_field: order} if order else None,
    )
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="",
        yaxis_title="Tokens",
        yaxis=dict(gridcolor="#f0f0f0"),
        legend_title="Component",
        height=370,
        bargap=0.1,
    )
    return fig


# ---------------------------------------------------------------------------
# Auto-refresh: bust data caches only when new rows arrive
# ---------------------------------------------------------------------------

latest_ts = _get_latest_timestamp()
if "last_ts" not in st.session_state:
    st.session_state.last_ts = latest_ts
if latest_ts != st.session_state.last_ts:
    st.session_state.last_ts = latest_ts
    load_runs.clear()
    load_cases.clear()
    load_l1_runs.clear()
    load_l1_cases.clear()

st_autorefresh(interval=POLL_SECONDS * 1000, key="data_poll")

# ---------------------------------------------------------------------------
# Guard: no database / no runs yet
# ---------------------------------------------------------------------------

if not DB_PATH.exists():
    st.warning(
        f"Database not found at `{DB_PATH}`.  \n"
        "Run the evaluator first:  \n"
        "`python eval_runner.py --layer 2`"
    )
    st.stop()

runs_df     = load_runs()
cases_df    = load_cases()
l1_runs_df  = load_l1_runs()
l1_cases_df = load_l1_cases()
sessions_df = build_sessions(l1_runs_df, runs_df)

if sessions_df.empty:
    st.info("No evaluation sessions recorded yet. Run the evaluator first.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — session selector
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Evaluation Ledger")
    st.caption(f"Database: `{DB_PATH.name}`")
    st.metric("Total sessions", len(sessions_df))

    def _session_label(sid: str) -> str:
        row = sessions_df[sessions_df["session_id"] == sid].iloc[0]
        ts  = row["run_timestamp"].strftime("%Y-%m-%d %H:%M")
        return f"{ts}  ·  {' + '.join(row['variants'])}"

    selected_session_id = st.selectbox(
        "Select session",
        options=sessions_df["session_id"].tolist(),
        format_func=_session_label,
    )

    session_row = sessions_df[sessions_df["session_id"] == selected_session_id].iloc[0]
    _idx = sessions_df.index[sessions_df["session_id"] == selected_session_id][0]
    prev_session_row = (
        sessions_df.iloc[_idx + 1] if _idx + 1 < len(sessions_df) else None
    )

    # All ledger rows for the selected session (L1 + L2 variants).
    session_l1_runs = (
        l1_runs_df[l1_runs_df["session_id"] == selected_session_id]
        if not l1_runs_df.empty else pd.DataFrame()
    )
    session_l2_runs = runs_df[runs_df["session_id"] == selected_session_id]
    session_all_runs = pd.concat(
        [df for df in (session_l1_runs, session_l2_runs) if not df.empty],
        ignore_index=True,
    )

    # Gate thresholds are written identically to every L2 variant in a session,
    # but L1 rows store NULL (no judge / no judge threshold). Prefer an L2 row
    # so threshold display doesn't blank out for sessions that include L1.
    rep_run_row = (
        session_l2_runs.iloc[0] if not session_l2_runs.empty
        else session_all_runs.iloc[0] if not session_all_runs.empty
        else None
    )
    _run_gate_thresholds = (
        _parse_json_col(rep_run_row.get("gate_thresholds"))
        if rep_run_row is not None else {}
    )
    _gates_md = "  \n".join(
        f"&nbsp;&nbsp;• **{k}**: {_run_gate_thresholds.get(k, NOT_EVALUATED)}"
        for k in EvaluationEngine.GATE_THRESHOLD_KEYS
    )

    _ordered = list(session_row["variants"])
    _mode_md = "  \n".join(f"&nbsp;&nbsp;• {v}" for v in _ordered)

    # Per-variant case counts. Fixture mode can run fewer cases than live
    # (it skips cases without a populated 'fixture' block), so collapsing
    # variants into one number could hide a coverage gap.
    _counts_by_variant: dict[str, int] = {}
    for _, _r in session_all_runs.iterrows():
        _label = _variant_label(_r.get("layer"), _r.get("execution_mode"))
        _cc    = _r.get("case_count")
        if _cc is not None and not pd.isna(_cc):
            _counts_by_variant[_label] = int(_cc)
    _cases_md = "  \n".join(
        f"&nbsp;&nbsp;• **{v}**: {_counts_by_variant[v]}"
        for v in _ordered if v in _counts_by_variant
    )

    st.divider()
    st.caption("Session details")
    st.markdown(
        f"**Judge:** {session_row.get('judge_model') or NOT_EVALUATED}  \n"
        f"**Mode:**  \n{_mode_md}  \n"
        f"**Cases:**  \n{_cases_md}  \n"
        f"**Thresholds:**  \n{_gates_md}"
    )

# Resolve the rep row of the previous session for the drift warning. Prefer
# an L2 row since L1 rows store NULL gate_thresholds.
if prev_session_row is not None:
    _prev_sid = prev_session_row["session_id"]
    _prev_l1  = (
        l1_runs_df[l1_runs_df["session_id"] == _prev_sid]
        if not l1_runs_df.empty else pd.DataFrame()
    )
    _prev_l2  = runs_df[runs_df["session_id"] == _prev_sid]
    prev_run_row = (
        _prev_l2.iloc[0] if not _prev_l2.empty
        else _prev_l1.iloc[0] if not _prev_l1.empty
        else None
    )
else:
    prev_run_row = None

# Gate-threshold drift warning (compare current session vs previous session).
gate_thresholds = (
    _parse_json_col(rep_run_row.get("gate_thresholds"))
    if rep_run_row is not None else {}
)
prev_gate_thresholds = (
    _parse_json_col(_prev(prev_run_row, "gate_thresholds"))
    if prev_run_row is not None else {}
)

# Per-session score bands. Falls back to the eval_config default when the
# session predates gate_thresholds being recorded (older L1 rows store
# NULL) or when the JSON column failed to parse.
session_judge_threshold = float(
    gate_thresholds.get("judge_threshold", _DEFAULT_JUDGE_THRESHOLD)
)
SCORE_BANDS = _score_bands(session_judge_threshold)

# ---------------------------------------------------------------------------
# Title and tabs
# ---------------------------------------------------------------------------

st.title("Evaluation Dashboard")
st.caption(
    "CompetitiveIntelligenceAgent + Deterministic Metrics"
)

if gate_thresholds and prev_gate_thresholds and gate_thresholds != prev_gate_thresholds:
    drift = [
        f"**{k}**: {prev_gate_thresholds.get(k)} → {gate_thresholds.get(k)}"
        for k in sorted(set(gate_thresholds) | set(prev_gate_thresholds))
        if gate_thresholds.get(k) != prev_gate_thresholds.get(k)
    ]
    st.warning(
        "⚠️ Gate thresholds changed since the previous run — "
        "trend comparisons may not be apples-to-apples.\n\n"
        + "\n\n".join(f"- {line}" for line in drift)
    )

tab_summary, tab_analysis, tab_trends, tab_guide = st.tabs([
    "Run Summary", "Deep Analysis", "Historical Trends", "Metrics Guide",
])


# ===========================================================================
# Tab 1 — Run Summary
# ===========================================================================

with tab_summary:

    # Per-variant ledger rows for the selected session. Each variant lives
    # in its own row in eval_runs (l1 in l1_runs_df, l2_fixture/l2_live in
    # runs_df) and any of them may be missing if the user didn't include
    # that variant in RUN_VARIANTS for this session.
    session_l1_run = (
        session_l1_runs.iloc[0] if not session_l1_runs.empty else None
    )
    _fx_runs = session_l2_runs[session_l2_runs["execution_mode"] == "fixture"]
    session_fixture_run = _fx_runs.iloc[0] if not _fx_runs.empty else None
    _lv_runs = session_l2_runs[session_l2_runs["execution_mode"] == "live"]
    session_live_run = _lv_runs.iloc[0] if not _lv_runs.empty else None

    # Per-variant case frames for the selected session.
    l1_run_cases = (
        l1_cases_df[l1_cases_df["run_id"] == int(session_l1_run["run_id"])]
        if session_l1_run is not None and not l1_cases_df.empty
        else pd.DataFrame()
    )
    fixture_cases = (
        cases_df[cases_df["run_id"] == int(session_fixture_run["run_id"])]
        if session_fixture_run is not None else pd.DataFrame()
    )
    live_cases_table = (
        cases_df[cases_df["run_id"] == int(session_live_run["run_id"])]
        if session_live_run is not None else pd.DataFrame()
    )

    def _mean(df: pd.DataFrame, col: str) -> float | None:
        if col not in df.columns:
            return None
        series = pd.to_numeric(df[col], errors="coerce")
        return None if series.dropna().empty else float(series.mean())

    # --- Average Performance Over Cases — L1 ---
    st.subheader("Average Performance Over Cases — L1")
    if session_l1_run is None:
        st.info("L1 was not run in this session.")
    else:
        l1_in  = _mean(l1_run_cases, "agent_input_tokens")
        l1_out = _mean(l1_run_cases, "agent_output_tokens")
        l1_total_tokens = (l1_in or 0) + (l1_out or 0) if (l1_in is not None or l1_out is not None) else None
        sa_row1 = st.columns(7)
        sa_row1[0].metric(LABELS["pass_rate"],                 fmt_for("pass_rate", session_l1_run.get("pass_rate")))
        sa_row1[1].metric(LABELS["latency_seconds"],           fmt_for("latency_seconds", _mean(l1_run_cases, "latency_seconds"), suffix="s"))
        sa_row1[2].metric(LABELS["agent_total_tokens"],        fmt_for("agent_total_tokens", l1_total_tokens))
        sa_row1[3].metric(LABELS["domain_inclusion_present"],  fmt_for("domain_inclusion_present", _mean(l1_run_cases, "domain_inclusion_present")))
        sa_row1[4].metric(LABELS["domain_exclusion_respected"],fmt_for("domain_exclusion_respected", _mean(l1_run_cases, "domain_exclusion_respected")))
        sa_row1[5].metric(LABELS["required_keyword_hit_rate"], fmt_for("required_keyword_hit_rate", _mean(l1_run_cases, "required_keyword_hit_rate")))
        sa_row1[6].metric(LABELS["disallowed_keyword_hits"],   fmt_for("disallowed_keyword_hits", _mean(l1_run_cases, "disallowed_keyword_hits")))
        sa_row2 = st.columns(7)
        sa_row2[0].metric(LABELS["tool_call_count"],                fmt_for("avg_tool_call_count", session_l1_run.get("avg_tool_call_count")))
        sa_row2[1].metric(LABELS["avg_recursion_limit_hit_rate"],   fmt_for("avg_recursion_limit_hit_rate", session_l1_run.get("avg_recursion_limit_hit_rate")))

    # --- Average Performance Over Cases — L2 fixture ---
    st.subheader("Average Performance Over Cases — L2 fixture")
    if session_fixture_run is None:
        st.info("L2 fixture was not run in this session.")
    else:
        ma = _parse_json_col(session_fixture_run.get("metric_averages"))
        fx_row1 = st.columns(7)
        fx_row1[0].metric(LABELS["pass_rate"],                  fmt_for("pass_rate", session_fixture_run.get("pass_rate")))
        fx_row1[1].metric(LABELS["avg_judge_run_score"],        fmt_for("avg_judge_run_score", session_fixture_run.get("avg_judge_run_score")))
        fx_row1[2].metric(LABELS["required_sections_present"],  fmt_for("required_sections_present", ma.get("required_sections_present")))
        fx_row1[3].metric(LABELS["citation_presence"],          fmt_for("citation_presence", ma.get("citation_presence")))
        fx_row1[4].metric(LABELS["significance_score_valid"],   fmt_for("avg_significance_score_valid", session_fixture_run.get("avg_significance_score_valid")))
        fx_row1[5].metric(LABELS["answer_relevancy"],           fmt_for("answer_relevancy", ma.get("answer_relevancy")))
        fx_row1[6].metric(LABELS["faithfulness"],               fmt_for("faithfulness", ma.get("faithfulness")))
        fx_row2 = st.columns(7)
        fx_row2[0].metric(LABELS["no_speculation"],             fmt_for("no_speculation", ma.get("no_speculation")))
        fx_row2[1].metric(LABELS["delta_quality"],              fmt_for("delta_quality", ma.get("delta_quality")))
        fx_row2[2].metric(LABELS["significance_justification"], fmt_for("significance_justification", ma.get("significance_justification")))
        fx_row2[3].metric(LABELS["required_keyword_hit_rate"],  fmt_for("avg_required_keyword_hit_rate", session_fixture_run.get("avg_required_keyword_hit_rate")))
        fx_row2[4].metric(LABELS["disallowed_keyword_hits"],    fmt_for("avg_disallowed_keyword_hits", session_fixture_run.get("avg_disallowed_keyword_hits")))
        fx_row2[5].metric(LABELS["judge_total_tokens"],         fmt_for("avg_judge_total_tokens", session_fixture_run.get("avg_judge_total_tokens")))

    # --- Average Performance Over Cases — L2 live ---
    st.subheader("Average Performance Over Cases — L2 live")
    if session_live_run is None:
        st.info("L2 live was not run in this session.")
    else:
        live_in  = _mean(live_cases_table, "agent_input_tokens")
        live_out = _mean(live_cases_table, "agent_output_tokens")
        live_total_tokens = (live_in or 0) + (live_out or 0) if (live_in is not None or live_out is not None) else None
        ma_live = _parse_json_col(session_live_run.get("metric_averages"))
        live_row1 = st.columns(8)
        live_row1[0].metric(LABELS["pass_rate"],                  fmt_for("pass_rate", session_live_run.get("pass_rate")))
        live_row1[1].metric(LABELS["avg_judge_run_score"],        fmt_for("avg_judge_run_score", session_live_run.get("avg_judge_run_score")))
        live_row1[2].metric(LABELS["latency_seconds"],            fmt_for("latency_seconds", _mean(live_cases_table, "latency_seconds"), suffix="s"))
        live_row1[3].metric(LABELS["agent_total_tokens"],         fmt_for("agent_total_tokens", live_total_tokens))
        live_row1[4].metric(LABELS["required_sections_present"],  fmt_for("required_sections_present", ma_live.get("required_sections_present")))
        live_row1[5].metric(LABELS["citation_presence"],          fmt_for("citation_presence", ma_live.get("citation_presence")))
        live_row1[6].metric(LABELS["domain_exclusion_respected"], fmt_for("avg_domain_exclusion_respected", session_live_run.get("avg_domain_exclusion_respected")))
        live_row1[7].metric(LABELS["answer_relevancy"],           fmt_for("answer_relevancy", ma_live.get("answer_relevancy")))
        live_row2 = st.columns(7)
        live_row2[0].metric(LABELS["faithfulness"],                  fmt_for("faithfulness", ma_live.get("faithfulness")))
        live_row2[1].metric(LABELS["no_speculation"],                fmt_for("no_speculation", ma_live.get("no_speculation")))
        live_row2[2].metric(LABELS["required_keyword_hit_rate"],     fmt_for("avg_required_keyword_hit_rate", session_live_run.get("avg_required_keyword_hit_rate")))
        live_row2[3].metric(LABELS["disallowed_keyword_hits"],       fmt_for("avg_disallowed_keyword_hits", session_live_run.get("avg_disallowed_keyword_hits")))
        live_row2[4].metric(LABELS["directive_compliance"],          fmt_for("avg_directive_compliance", session_live_run.get("avg_directive_compliance")))
        live_row2[5].metric(LABELS["tool_call_count"],               fmt_for("avg_tool_call_count", session_live_run.get("avg_tool_call_count")))
        live_row2[6].metric(LABELS["avg_recursion_limit_hit_rate"],  fmt_for("avg_recursion_limit_hit_rate", session_live_run.get("avg_recursion_limit_hit_rate")))
        live_row3 = st.columns(7)
        live_row3[0].metric(LABELS["judge_total_tokens"],            fmt_for("avg_judge_total_tokens", session_live_run.get("avg_judge_total_tokens")))

    st.divider()

    st.subheader("Per-Case Results")
    st.caption("**Color key for the tables below:**")
    st.markdown(
        "- :green[**green**] = passing comfortably (score ≥ midpoint of judge threshold and 1.0)\n"
        "- :orange[**orange**] = borderline (≥ threshold but below the green band)\n"
        "- :red[**red**] = failing (< threshold)\n"
        "- :gray[**gray**] = NULL / not evaluated"
    )
    st.caption(
        "The green/orange split scales with the run's judge threshold — a 0.6 score is borderline at threshold 0.5 but failing at 0.7. "
        "Latency uses fixed bands instead: green ≤ 5s, orange ≤ 10s, red > 10s."
    )

    # --- Per-case tables: L1, L2 fixture, L2 live ---
    def _render_l2_case_table(
        cases: pd.DataFrame,
        label: str,
        *,
        exclude: set[str] | None = None,
    ) -> None:
        """Render the per-case results table for an L2 variant.

        exclude skips specific columns from being displayed without
        removing them from the underlying data — used for fixture mode where
        tool_call_count, directive_compliance, and latency_seconds
        are structurally NULL (the agent LLM doesn't run) and would only add
        noise to the table.
        """
        if cases.empty:
            st.info(f"No L2 {label} cases recorded yet.")
            return
        exclude = exclude or set()
        score_cols = [
            k for k in JUDGE_KEYS + FORMAT_KEYS + ROUTING_KEYS + DELTA_KEYS
            if k in cases.columns and k not in exclude
        ]
        token_cols = [
            k for k in JUDGE_TOKEN_KEYS
            if k in cases.columns and k not in exclude
        ]
        display_cols = ["case_id", "layer", "category", "status"] + score_cols + token_cols
        if "latency_seconds" not in exclude:
            display_cols.append("latency_seconds")
        available = [c for c in display_cols if c in cases.columns]
        table_df  = cases[available].reset_index(drop=True)

        col_config: dict[str, Any] = {}
        if "case_id" in table_df.columns:
            col_config["case_id"] = st.column_config.TextColumn(
                label=LABELS.get("case_id", "case_id"), pinned=True
            )
        for col in ("layer", "category", "status"):
            if col in table_df.columns:
                col_config[col] = st.column_config.Column(label=LABELS.get(col, col))
        for key in score_cols + token_cols:
            if key in table_df.columns:
                col_config[key] = st.column_config.NumberColumn(
                    label=LABELS.get(key, key), format=column_format(key)
                )
        if "latency_seconds" in table_df.columns:
            col_config["latency_seconds"] = st.column_config.NumberColumn(
                label=LABELS["latency_seconds"],
                format=column_format("latency_seconds", suffix="s"),
            )

        score_subset = [c for c in score_cols if c in table_df.columns]
        styler = table_df.style
        if score_subset:
            styler = styler.map(lambda v: _color_by_thresholds(v, SCORE_BANDS), subset=score_subset)
        if "latency_seconds" in table_df.columns:
            styler = styler.map(
                lambda v: _color_by_thresholds(v, _LATENCY_BANDS, higher_is_better=False),
                subset=["latency_seconds"],
            )

        st.dataframe(
            styler,
            use_container_width=True,
            height=min(len(table_df) * 38 + 44, 520),
            column_config=col_config,
        )

    # --- Per-Case Results — L1 ---
    st.subheader("Per-Case Results — L1")
    if l1_run_cases.empty:
        st.info("No L1 cases recorded yet.")
    else:
        l1_score_cols = [
            c for c in [
                "result_count",
                "domain_exclusion_respected",
                "domain_inclusion_present",
                "latency_seconds",
            ]
            if c in l1_run_cases.columns
        ]
        l1_display_cols = ["case_id", "category", "status"] + l1_score_cols
        l1_available = [c for c in l1_display_cols if c in l1_run_cases.columns]
        l1_table_df = l1_run_cases[l1_available].reset_index(drop=True)

        l1_col_config: dict[str, Any] = {}
        if "case_id" in l1_table_df.columns:
            l1_col_config["case_id"] = st.column_config.TextColumn(
                label=LABELS.get("case_id", "case_id"), pinned=True
            )
        for col in ("category", "status"):
            if col in l1_table_df.columns:
                l1_col_config[col] = st.column_config.Column(label=LABELS.get(col, col))
        for key in ("result_count", "domain_exclusion_respected", "domain_inclusion_present"):
            if key in l1_table_df.columns:
                l1_col_config[key] = st.column_config.NumberColumn(
                    label=LABELS[key], format=column_format(key)
                )
        if "latency_seconds" in l1_table_df.columns:
            l1_col_config["latency_seconds"] = st.column_config.NumberColumn(
                label=LABELS["latency_seconds"],
                format=column_format("latency_seconds", suffix="s"),
            )

        l1_score_subset = [
            c for c in ("domain_exclusion_respected", "domain_inclusion_present")
            if c in l1_table_df.columns
        ]
        l1_styler = l1_table_df.style
        if l1_score_subset:
            l1_styler = l1_styler.map(
                lambda v: _color_by_thresholds(v, SCORE_BANDS),
                subset=l1_score_subset,
            )

        st.dataframe(
            l1_styler,
            use_container_width=True,
            height=min(len(l1_table_df) * 38 + 44, 360),
            column_config=l1_col_config,
        )

    # --- Per-Case Results — L2 fixture ---
    st.subheader("Per-Case Results — L2 fixture")
    _render_l2_case_table(
        fixture_cases,
        "fixture",
        exclude={"tool_call_count", "directive_compliance", "latency_seconds"},
    )

    # --- Per-Case Results — L2 live ---
    st.subheader("Per-Case Results — L2 live")
    _render_l2_case_table(
        live_cases_table,
        "live",
        exclude={
            "delta_quality",
            "significance_justification",
            "significance_score",
            "score_in_expected_range",
        },
    )

    # --- Answer details: L1, L2 fixture, L2 live ---
    def _render_answer_detail(
        cases: pd.DataFrame,
        empty_msg: str,
        answer_label: str,
        right_rail: Callable[[pd.Series], None],
        title_suffix: Callable[[pd.Series], str] = lambda _cr: "",
    ) -> None:
        if cases.empty:
            st.info(empty_msg)
            return
        for _, cr in cases.iterrows():
            icon = "✅" if cr.get("status") == "PASS" else "⚠️"
            with st.expander(f"{icon}  {cr['case_id']}  —  {cr.get('status', '?')}{title_suffix(cr)}"):
                left_col, right_col = st.columns([3, 1])
                with left_col:
                    st.markdown("**Question:**")
                    with st.container(border=True):
                        st.markdown(str(cr.get("question") or NOT_EVALUATED))
                    st.markdown(f"**{answer_label}:**")
                    # Demote any markdown headings in the answer (#, ##, ###)
                    # so they don't render larger than the section labels above.
                    answer_md = re.sub(
                        r"(?m)^(#{1,3})\s",
                        lambda m: "#" * (len(m.group(1)) + 3) + " ",
                        str(cr.get("answer") or NOT_EVALUATED),
                    )
                    with st.container(border=True):
                        st.markdown(answer_md)
                with right_col:
                    right_rail(cr)
                    if cr.get("errors") and cr["errors"] != "[]":
                        st.error(f"Errors: {cr['errors']}")

    def _l2_fixture_right_rail(cr: pd.Series) -> None:
        sig = cr.get("significance_score")
        st.metric(LABELS["status"], cr.get("status", NOT_EVALUATED))
        st.metric(LABELS["avg_judge_score"], fmt_for("avg_judge_score", cr.get("avg_judge_score")))
        st.metric(
            LABELS["significance_score"],
            str(int(sig)) if (sig is not None and pd.notna(sig)) else NOT_EVALUATED,
        )

    def _l2_live_right_rail(cr: pd.Series) -> None:
        st.metric(LABELS["status"], cr.get("status", NOT_EVALUATED))
        st.metric(LABELS["avg_judge_score"], fmt_for("avg_judge_score", cr.get("avg_judge_score")))
        st.metric(LABELS["latency_seconds"], fmt_for("latency_seconds", cr.get("latency_seconds"), suffix="s"))

    def _l1_right_rail(cr: pd.Series) -> None:
        rc = cr.get("result_count")
        st.metric(LABELS["status"], cr.get("status", NOT_EVALUATED))
        st.metric(LABELS["result_count"], str(int(rc)) if pd.notna(rc) and rc is not None else NOT_EVALUATED)
        st.metric(LABELS["domain_inclusion_present"], fmt_for("domain_inclusion_present", cr.get("domain_inclusion_present")))
        st.metric(LABELS["domain_exclusion_respected"], fmt_for("domain_exclusion_respected", cr.get("domain_exclusion_respected")))
        st.metric(LABELS["latency_seconds"], fmt_for("latency_seconds", cr.get("latency_seconds"), suffix="s"))

    st.divider()
    st.subheader("Answer Detail — L1")
    _render_answer_detail(l1_run_cases, "No L1 cases recorded yet.", "Sub-agent answer", _l1_right_rail)

    st.divider()
    st.subheader("Answer Detail — L2 fixture")
    _render_answer_detail(fixture_cases, "No L2 fixture cases recorded yet.",
                          "Agent answer", _l2_fixture_right_rail)

    st.divider()
    st.subheader("Answer Detail — L2 live")
    _render_answer_detail(live_cases_table, "No L2 live cases recorded yet.",
                          "Agent answer", _l2_live_right_rail)


# ===========================================================================
# Tab 2 — Deep Analysis
# ===========================================================================

with tab_analysis:

    st.caption("Deep analysis of metrics for the selected session.")

    # Build a session-wide cases frame so charts cover L1 + L2 fixture + L2
    # live together. The sidebar already resolved session_l1_runs and
    # session_l2_runs for the selected session.
    _l1_session_run_ids = session_l1_runs["run_id"].tolist() if not session_l1_runs.empty else []
    _l2_session_run_ids = session_l2_runs["run_id"].tolist() if not session_l2_runs.empty else []
    l1_session_cases = (
        l1_cases_df[l1_cases_df["run_id"].isin(_l1_session_run_ids)].copy()
        if _l1_session_run_ids and not l1_cases_df.empty else pd.DataFrame()
    )
    l2_session_cases = (
        cases_df[cases_df["run_id"].isin(_l2_session_run_ids)].copy()
        if _l2_session_run_ids else pd.DataFrame()
    )
    session_cases = pd.concat([l1_session_cases, l2_session_cases], ignore_index=True)

    if session_cases.empty:
        st.warning("No case-level data for this session.")
    else:
        # All numeric metrics that make sense for a histogram, in canonical
        # reading order. Every list is registry-derived so adding a metric
        # (e.g. a new token group) flows through automatically. Filtered
        # below to keys with at least one non-null value in this session.
        candidate_metrics = (
            JUDGE_KEYS                   # 0-1 LLM-judged scores
            + FORMAT_KEYS                # 0-1 format gates
            + DELTA_KEYS                 # significance_score (1-10), score_in_expected_range (0/1)
            + ROUTING_KEYS               # directive_compliance (0/1), tool_call_count (int)
            + KEYWORD_KEYS               # required_keyword_hit_rate (0-1), disallowed_keyword_hits (int)
            + SUBAGENT_KEYS              # domain checks (0/1), result_count (int)
            + LATENCY_KEYS               # latency_seconds (float)
            + TOKEN_KEYS                 # agent_input/output/total_tokens (int)
            + JUDGE_TOKEN_KEYS           # judge_input/output/total_tokens (int)
            + JUDGE_AGGREGATE_KEYS       # avg_judge_score (0-1)
            + DIAGNOSTIC_KEYS            # recursion_limit_hit (0/1), report_length (int)
        )
        all_available = [
            m for m in candidate_metrics
            if m in session_cases.columns
            and not pd.to_numeric(session_cases[m], errors="coerce").dropna().empty
        ]

        # Metrics on a known [0, 1] scale. Everything else either has a
        # known higher max (significance_score is 1-10) or is unbounded
        # (latency, tokens, integer counts) and should auto-fit to data.
        NORMALIZED_0_1_METRICS = set(
            JUDGE_KEYS
            + FORMAT_KEYS
            + ["directive_compliance", "score_in_expected_range",
               "required_keyword_hit_rate", "domain_exclusion_respected",
               "domain_inclusion_present", "recursion_limit_hit", "avg_judge_score"]
        )

        # ── Score distribution ─────────────────────────────────────────────
        st.subheader("Score Distribution")
        st.caption("Distribution across all cases in this session (L1 + L2 fixture + L2 live).")
        if all_available:
            selected_dist_metric = st.selectbox(
                "Metric",
                options=all_available,
                format_func=lambda c: LABELS.get(c, c),
                label_visibility="collapsed",
                key="deep_dist_metric",
            )
            dist_series = pd.to_numeric(session_cases[selected_dist_metric], errors="coerce").dropna()
            if not dist_series.empty:
                fig_dist = px.histogram(
                    dist_series.to_frame(name=selected_dist_metric),
                    x=selected_dist_metric,
                    nbins=10,
                    color_discrete_sequence=["#636EFA"],
                )
                if selected_dist_metric in NORMALIZED_0_1_METRICS:
                    x_range = [0, 1.05]
                elif selected_dist_metric == "significance_score":
                    x_range = [0, 10.5]
                else:
                    x_range = None  # auto-fit for unbounded metrics
                layout_kwargs: dict[str, Any] = dict(
                    PLOTLY_LAYOUT_DEFAULTS,
                    xaxis_title=LABELS.get(selected_dist_metric, selected_dist_metric),
                    yaxis_title="Cases",
                    height=400,
                )
                if x_range is not None:
                    layout_kwargs["xaxis"] = dict(range=x_range)
                fig_dist.update_layout(**layout_kwargs)
                st.plotly_chart(fig_dist, key="deep_histogram", use_container_width=True)
            else:
                st.info("No data for this metric.")

        st.divider()

        # Live run_ids in the selected session. Used by both the latency
        # and token charts to filter L2 cases down to live-mode rows
        # (fixture rows carry no real wall-clock or token usage).
        _live_run_ids = (
            session_l2_runs.query("execution_mode == 'live'")["run_id"].tolist()
            if "execution_mode" in session_l2_runs.columns else []
        )

        # ── Latency breakdown ──────────────────────────────────────────────
        st.subheader("Latency Breakdown by Case")
        st.caption("Per-case wall-clock latency, grouped by L1 and L2 live (fixture has no live latency).")
        l1_lat = l1_session_cases.assign(_layer_group="L1") if not l1_session_cases.empty else pd.DataFrame()
        if not l2_session_cases.empty and _live_run_ids:
            l2_live_lat = (
                l2_session_cases[l2_session_cases["run_id"].isin(_live_run_ids)]
                .assign(_layer_group="L2 live")
            )
        else:
            l2_live_lat = pd.DataFrame()
        latency_source = pd.concat([l1_lat, l2_live_lat], ignore_index=True)
        fig_latency = build_latency_bar(latency_source, group_col="_layer_group")
        if fig_latency is not None:
            st.plotly_chart(fig_latency, key="deep_latency_bar", use_container_width=True)
        else:
            st.info("No latency data recorded for this session.")

        st.divider()

        # ── Token usage ───────────────────────────────────────────────
        st.subheader("Token Usage by Case")
        st.caption("Input vs output tokens, grouped by L1 and L2 live (fixture has no live tokens).")
        l1_tok = l1_session_cases.assign(_layer_group="L1") if not l1_session_cases.empty else pd.DataFrame()
        if not l2_session_cases.empty and _live_run_ids:
            l2_live_tok = (
                l2_session_cases[l2_session_cases["run_id"].isin(_live_run_ids)]
                .assign(_layer_group="L2 live")
            )
        else:
            l2_live_tok = pd.DataFrame()
        token_source = pd.concat([l1_tok, l2_live_tok], ignore_index=True)
        fig_tokens = build_token_bar(token_source, group_col="_layer_group")
        if fig_tokens is not None:
            st.plotly_chart(fig_tokens, key="deep_token_bar", use_container_width=True)
        else:
            st.info("No token usage recorded for this session.")



# ===========================================================================
# Tab 3 — Historical Trends
# ===========================================================================

with tab_trends:

    st.subheader("Historical Trends")

    l1_runs_trend  = load_l1_runs()
    l1_cases_trend = load_l1_cases()

    l2_fixture_runs = (
        runs_df[runs_df.get("execution_mode") == "fixture"].copy()
        if "execution_mode" in runs_df.columns else pd.DataFrame()
    )
    l2_live_runs = (
        runs_df[runs_df.get("execution_mode") == "live"].copy()
        if "execution_mode" in runs_df.columns else pd.DataFrame()
    )

    def _series(df: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
        if df.empty or value_col not in df.columns:
            return pd.DataFrame(columns=["run_timestamp", "value", "series"])
        out = df[["run_timestamp", value_col]].copy()
        out = out.rename(columns={value_col: "value"})
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out["series"] = label
        return out.dropna(subset=["value"])

    def _tokens_series(runs: pd.DataFrame, cases: pd.DataFrame, label: str) -> pd.DataFrame:
        if runs.empty or cases.empty or "run_id" not in cases.columns:
            return pd.DataFrame(columns=["run_timestamp", "value", "series"])
        c = cases.copy()
        c["total_tokens"] = (
            pd.to_numeric(c.get("agent_input_tokens"),  errors="coerce").fillna(0)
            + pd.to_numeric(c.get("agent_output_tokens"), errors="coerce").fillna(0)
        )
        per_run = c.groupby("run_id", as_index=False)["total_tokens"].mean()
        merged = runs[["run_id", "run_timestamp"]].merge(per_run, on="run_id", how="inner")
        if merged.empty:
            return pd.DataFrame(columns=["run_timestamp", "value", "series"])
        merged = merged.rename(columns={"total_tokens": "value"})
        merged["series"] = label
        return merged[["run_timestamp", "value", "series"]]

    def _line(df: pd.DataFrame, title: str, empty_msg: str, y_label: str, x_label: str = "Time") -> None:
        st.markdown(f"**{title}**")
        if df.empty:
            st.info(empty_msg)
            return
        df = df.sort_values("run_timestamp")
        st.line_chart(df, x="run_timestamp", y="value", color="series",
                      x_label=x_label, y_label=y_label,
                      use_container_width=True)

    # --- Chart 1: Pass Rate ---
    pass_df = pd.concat([
        _series(l1_runs_trend,  "pass_rate", "L1"),
        _series(l2_fixture_runs, "pass_rate", "L2 fixture"),
        _series(l2_live_runs,    "pass_rate", "L2 live"),
    ], ignore_index=True)
    _line(pass_df, "Pass Rate Over Time", "No pass rate history yet.", "Pass Rate")

    # --- Chart 2: Quality scores ---
    score_df = pd.concat([
        _series(l1_runs_trend,   "avg_required_keyword_hit_rate", "L1 Keyword Hit Rate"),
        _series(l2_fixture_runs, "avg_judge_run_score",           "L2 Fixture Avg Judge Score"),
        _series(l2_live_runs,    "avg_judge_run_score",            "L2 Live Avg Judge Score"),
    ], ignore_index=True)
    _line(score_df, "Quality Scores Over Time", "No quality score history yet.", "Score")

    # --- Chart 3: Latency ---
    latency_df = pd.concat([
        _series(l1_runs_trend, "avg_latency_seconds", "L1"),
        _series(l2_live_runs,  "avg_latency_seconds", "L2 live"),
    ], ignore_index=True)
    _line(latency_df, "Latency Over Time", "No latency history yet.", "Latency (seconds)")

    # --- Chart 4: Tokens ---
    tokens_df = pd.concat([
        _tokens_series(l1_runs_trend, l1_cases_trend, "L1"),
        _tokens_series(l2_live_runs,  cases_df,       "L2 live"),
    ], ignore_index=True)
    _line(tokens_df, "Avg Tokens per Case Over Time", "No token history yet.", "Tokens")


# ===========================================================================
# Tab 5 — Metrics Guide
# ===========================================================================

with tab_guide:
    # Render info_metrics.md directly so this tab never drifts from the
    # source-of-truth doc. Reading inside the tab body (not at module
    # level) means markdown edits show up on the next Streamlit rerun
    # without restarting the dashboard.
    #
    # unsafe_allow_html=True is required so the doc's explicit
    # <a id="..."></a> jump targets work — the at-a-glance tables
    # link to per-metric sections, and Streamlit's auto-generated
    # heading anchors collide on duplicate-named sections (e.g.
    # "Sections Present" appears in both L2 fixture and L2 live).
    # The content is project-internal documentation, not user input.
    _guide_path = _EVAL_DIR / "info_metrics.md"
    if _guide_path.exists():
        st.markdown(_guide_path.read_text(encoding="utf-8"), unsafe_allow_html=True)
    else:
        st.warning(
            f"Metrics guide not found at `{_guide_path}`. "
            "Expected `info_metrics.md` alongside `eval_dashboard.py`."
        )
