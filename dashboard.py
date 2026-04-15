"""Streamlit dashboard for browsing saved competitive-intelligence results.

Run with:
    streamlit run dashboard.py --server.port 8501

The dashboard polls the SQLite ledger every POLL_SECONDS and refreshes
automatically when new rows are detected, without a full browser reload.
"""

# Standard library
import sqlite3
from pathlib import Path

# Third-party
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


POLL_SECONDS = 10  # How often to check if new data has arrived

DB_PATH = Path(__file__).resolve().parent / "agent_memory.db"
st.set_page_config(page_title="Strategic CI Dashboard", layout="wide")


@st.cache_data(ttl=POLL_SECONDS)
def _get_latest_timestamp() -> str | None:
    """Return the most recent Timestamp value from the ledger.

    Cached with a short TTL so the heavy ``load_data`` cache is only busted
    when a genuinely new row has arrived, not on every rerun.
    """
    if not DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            result = conn.execute("SELECT MAX(timestamp) FROM intel_ledger").fetchone()
            return result[0] if result else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the full intel ledger from SQLite into a DataFrame.

    Returns an empty DataFrame if the database file does not yet exist or if
    the query fails.  Cached indefinitely; call ``load_data.clear()`` to
    force a fresh read when new rows are detected.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()

    query = """
        SELECT Timestamp, Company, Significance, Importance, Description, Sentiment, Engine, Mode, Result
        FROM intel_ledger
    """

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Could not load dashboard data: {e}")
        return pd.DataFrame()

    if not df.empty and "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    return df


def color_significance(val):
    """Return red background CSS for rows flagged as strategic alerts (Importance == 1)."""
    return "background-color: #ff4b4b; color: white" if val == 1 else ""


# ---------------------------------------------------------------------------
# Auto-refresh: compare the latest DB timestamp to the previous run.
# If a new row arrived, bust the load_data cache to fetch fresh data.
# ---------------------------------------------------------------------------
latest_ts = _get_latest_timestamp()
if "last_ts" not in st.session_state:
    st.session_state.last_ts = latest_ts

if latest_ts != st.session_state.last_ts:
    st.session_state.last_ts = latest_ts
    load_data.clear()  # New data detected — force a fresh DB read

# Soft-rerun every POLL_SECONDS via a hidden Streamlit component.
# This avoids a full browser reload and prevents visible flicker.
st_autorefresh(interval=POLL_SECONDS * 1000, key="data_poll")

df = load_data()


# -- CHART PART --
st.title("Competitor Intelligence Dashboard")
st.caption("Browse recent competitive-intelligence findings stored in the local ledger.")

if df.empty:
    st.warning("No intelligence logs were found yet. Run the agent first to populate the dashboard.")
    st.stop()


# ---------------------------------------------------------------------------
# Sentiment-over-time chart
# Altair is imported lazily (inside the block) because it is slow to import
# and is only needed when there is data to visualise.
# ---------------------------------------------------------------------------
if "Timestamp" in df.columns and "Sentiment" in df.columns:
    import altair as alt

    chart_df = df[["Timestamp", "Company", "Sentiment"]].dropna().sort_values("Timestamp").copy()
    chart_df["Sentiment"] = pd.to_numeric(chart_df["Sentiment"], errors="coerce")
    all_companies = sorted(chart_df["Company"].dropna().unique().tolist())

    # Split screen into two side-by-side columns: left for the chart (80%), right for the company filter (20%).
    col_chart, col_filter = st.columns([4, 1])
    with col_filter:
        selected_companies = st.multiselect("Filter companies", options=all_companies, default=[])
    if selected_companies:
        chart_df = chart_df[chart_df["Company"].isin(selected_companies)]
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Timestamp:T", title="Time", axis=alt.Axis(tickCount="hour", format="%H:%M")),
            y=alt.Y("Sentiment:Q", title="Sentiment Score", scale=alt.Scale(domain=[0, 10])),
            color=alt.Color("Company:N", title="Company"),
        )
        .properties(height=250, title="Competitor Sentiment Over Time")
    )
    with col_chart:
        st.altair_chart(chart, use_container_width=True)


# Sidebar filters
st.sidebar.header("Filter Intel")
company_list = sorted(df["Company"].dropna().unique().tolist())
selected_company = st.sidebar.multiselect("Select Competitors", company_list, default=[])
show_only_alerts = st.sidebar.checkbox("Show only strategic alerts (7+)", value=False)
selected_engine = st.sidebar.multiselect("Select Engines", sorted(df["Engine"].dropna().unique().tolist()), default=[])
mode_list = sorted(df["Mode"].dropna().unique().tolist())
selected_mode = st.sidebar.multiselect("Select Mode", mode_list, default=[])

sort_by = st.sidebar.selectbox("Sort by", ["Timestamp", "Sentiment", "Significance"], index=0)
sort_order = st.sidebar.radio("Order", ["Descending", "Ascending"], index=0, horizontal=True)

filtered_df = df.copy()
if selected_company:
    filtered_df = filtered_df[filtered_df["Company"].isin(selected_company)]

if selected_engine:
    filtered_df = filtered_df[filtered_df["Engine"].isin(selected_engine)]

if selected_mode:
    filtered_df = filtered_df[filtered_df["Mode"].isin(selected_mode)]

if show_only_alerts:
    filtered_df = filtered_df[filtered_df["Importance"] == 1]

filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_order == "Ascending"), na_position="last")


# Key metrics summary boxes
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Intelligence Logs", len(df))
with col2:
    high_alerts = df[df["Importance"] == 1]
    st.metric("Strategic Alerts (7+)", len(high_alerts), delta_color="normal")
with col3:
    latest_update = df["Timestamp"].max() if not df.empty else "N/A"
    st.write(f"**Latest Sync:** {latest_update}")


# Intelligence ledger table with conditional formatting and detail view
st.subheader("▦ Intelligence Ledger")

if filtered_df.empty:
    st.info("No records match the current filters.")
else:
    table_df = filtered_df[
        ["Timestamp", "Company", "Significance", "Importance", "Description", "Sentiment", "Engine", "Mode"]
    ].copy()
    styler = table_df.style
    if hasattr(styler, "map"):
        styled_df = styler.map(color_significance, subset=["Importance"])
    else:
        styled_df = styler.applymap(color_significance, subset=["Importance"])
    st.dataframe(styled_df, width="stretch")

    st.divider()
    st.subheader("🔎 Deep-Dive Analysis")
    detail_options = filtered_df.index.tolist()
    selected_row = st.selectbox("Select a log to read the full report:", detail_options)

    if selected_row is not None:
        selected_record = filtered_df.loc[selected_row]
        st.markdown(f"### Report for {selected_record['Company']}")
        st.write(
            f"**Engine:** {selected_record['Engine']} | "
            f"**Significance:** {selected_record['Significance']} | "
            f"**Sentiment:** {selected_record['Sentiment']}"
        )
        st.markdown(selected_record["Result"])
