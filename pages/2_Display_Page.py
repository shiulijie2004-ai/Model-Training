import streamlit as st
import pandas as pd
import plotly.express as px

from core.config_user import COPYRIGHT_TEXT, PROJECT_TITLE

st.set_page_config(page_title="Display", layout="wide")
st.title(f"📊 Analysis: {PROJECT_TITLE}")
st.sidebar.markdown("---")
st.sidebar.info(COPYRIGHT_TEXT)

if "results" not in st.session_state or not st.session_state.results:
    st.warning("No results found. Train models first.")
    st.stop()

df = pd.DataFrame(st.session_state.results).copy()

# -------------------------------
# Helpers
# -------------------------------
CORE_METRICS = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)", "F1 (Weighted)"]
PER_CLASS_COLS = ["F1 (eating)", "F1 (ruminating)", "F1 (standing)", "F1 (walking)"]

def _to_numeric_cols(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def _numeric_columns(d: pd.DataFrame) -> list[str]:
    """Columns we should average when building 'average' rows."""
    numeric_candidates = CORE_METRICS + PER_CLASS_COLS
    # add optional profiler columns if exist
    optional = [
        "Training Time (ms)", "Latency (ms)", "Energy (mJ)", "Model Size (KB)",
        "Target Device",  # not numeric, but keep for first() later
    ]
    # Also average any other numeric columns that may exist
    _to_numeric_cols(d, numeric_candidates + ["Training Time (ms)", "Latency (ms)", "Energy (mJ)", "Model Size (KB)"])
    num_cols = []
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            num_cols.append(c)
    # Keep stable ordering: core+perclass first then others
    ordered = [c for c in numeric_candidates if c in num_cols] + [c for c in num_cols if c not in numeric_candidates]
    return ordered

def _make_avg_rows(d: pd.DataFrame, split_label: str = "loco_avg") -> pd.DataFrame:
    """Return one average row per algorithm (mean over numeric columns)."""
    if d.empty:
        return d

    num_cols = _numeric_columns(d)
    keep_cols = ["Algorithm"]
    if "Target Device" in d.columns:
        keep_cols.append("Target Device")

    # group mean
    g = d.groupby("Algorithm", as_index=False)
    avg = g[num_cols].mean(numeric_only=True)

    # add stable fields
    avg["Split"] = split_label
    avg["Test Group"] = "average"

    # attach first target device if present
    if "Target Device" in d.columns:
        first_dev = g["Target Device"].first().reset_index(drop=True)
        # align by Algorithm
        avg = avg.merge(first_dev, on="Algorithm", how="left")

    # column ordering
    front = ["Algorithm", "Split", "Test Group"]
    if "Target Device" in avg.columns:
        front.append("Target Device")
    rest = [c for c in avg.columns if c not in front]
    return avg[front + rest]

def _dynamic_y_range(values: pd.Series, pad: float = 0.05):
    """Dynamic y-range for 0..1 metrics so bars show variation."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return [0, 1]
    vmin, vmax = float(vals.min()), float(vals.max())
    if abs(vmax - vmin) < 1e-6:
        # avoid flat range
        lo = max(0.0, vmin - 0.1)
        hi = min(1.0, vmax + 0.1)
        return [lo, hi]
    lo = max(0.0, vmin - pad)
    hi = min(1.0, vmax + pad)
    # if still too tight, add a bit more
    if hi - lo < 0.15:
        mid = (hi + lo) / 2.0
        lo = max(0.0, mid - 0.1)
        hi = min(1.0, mid + 0.1)
    return [lo, hi]

def _aggregate_for_charts(d: pd.DataFrame, force_mean: bool = True) -> pd.DataFrame:
    """Ensure charts have ONE row per Algorithm (mean if multiple rows exist)."""
    if d.empty:
        return d
    if not force_mean:
        return d

    num_cols = _numeric_columns(d)
    g = d.groupby("Algorithm", as_index=False)
    out = g[num_cols].mean(numeric_only=True)
    out["Split"] = "aggregate"
    out["Test Group"] = "aggregate"

    if "Target Device" in d.columns:
        out = out.merge(g["Target Device"].first(), on="Algorithm", how="left")
    return out

# -------------------------------
# Basic sanity
# -------------------------------
if "Algorithm" not in df.columns:
    st.error("Result table has no 'Algorithm' column.")
    st.write("Available columns:", list(df.columns))
    st.stop()

# ensure columns exist
if "Split" not in df.columns:
    df["Split"] = "unknown"
if "Test Group" not in df.columns:
    df["Test Group"] = ""

# numeric conversion
df = _to_numeric_cols(df, CORE_METRICS + PER_CLASS_COLS + ["Training Time (ms)", "Latency (ms)", "Energy (mJ)", "Model Size (KB)"])

# -------------------------------
# Sidebar: View Mode
# -------------------------------
with st.sidebar:
    st.header("Display Options")

    has_loco = (df["Split"].astype(str).str.lower() == "loco").any()
    has_final = df["Split"].astype(str).str.lower().isin(["final", "all"]).any()

    mode_options = []
    if has_loco:
        mode_options.append("Full LOCO (cow-wise average)")
    if has_final:
        mode_options.append("Final deployment (train on ALL)")
    mode_options.append("Custom filter")

    view_mode = st.selectbox("View mode", mode_options, index=0)

    # algorithm filter always available
    algo_opts = sorted(df["Algorithm"].dropna().astype(str).unique().tolist())
    sel_algos = st.multiselect("Algorithm filter", options=algo_opts, default=algo_opts)

# -------------------------------
# Build view dataframe according to mode
# -------------------------------
df_view = df.copy()
df_view = df_view[df_view["Algorithm"].astype(str).isin(sel_algos)].copy()

# Full LOCO mode: only Split=loco folds (exclude final row), add average row
if view_mode == "Full LOCO (cow-wise average)":
    df_loco = df_view[df_view["Split"].astype(str).str.lower() == "loco"].copy()

    # allow choosing subset of cows
    tg_opts = sorted(df_loco["Test Group"].dropna().astype(str).unique().tolist())
    # default: all cows
    sel_tg = st.sidebar.multiselect("LOCO test groups", options=tg_opts, default=tg_opts)
    df_loco = df_loco[df_loco["Test Group"].astype(str).isin(sel_tg)].copy()

    df_avg = _make_avg_rows(df_loco, split_label="loco_avg")
    df_details = pd.concat([df_loco, df_avg], ignore_index=True)

    # charts should be based on average rows only (one per algo)
    df_chart = df_avg.copy()

elif view_mode == "Final deployment (train on ALL)":
    df_final = df_view[df_view["Split"].astype(str).str.lower().isin(["final", "all"])].copy()
    # if both exist, keep both but charts aggregate to one per algo
    df_details = df_final.copy()
    df_chart = _aggregate_for_charts(df_final, force_mean=True)

else:
    # Custom filter
    split_opts = sorted(df_view["Split"].dropna().astype(str).unique().tolist())
    sel_splits = st.sidebar.multiselect("Split filter", options=split_opts, default=split_opts)
    df_f = df_view[df_view["Split"].astype(str).isin(sel_splits)].copy()

    # If this includes multiple rows per algorithm (e.g., loco folds), aggregate for charts
    force_mean = st.sidebar.checkbox("Aggregate multiple rows per algorithm (mean)", value=True)
    df_details = df_f.copy()
    df_chart = _aggregate_for_charts(df_f, force_mean=force_mean)

# -------------------------------
# Metrics columns available
# -------------------------------
metrics = [c for c in CORE_METRICS if c in df_chart.columns]
if not metrics:
    st.error("No core metric columns found in results.")
    st.write("Available columns:", list(df.columns))
    st.stop()

tabs = st.tabs(["Charts", "Analysis", "Details"])
tab1, tab2, tab3 = tabs

# ======================================================
# TAB 1: Charts
# ======================================================
with tab1:
    st.subheader("Performance Comparison")

    # Melt charts table (already one row per algo)
    df_melt = df_chart[["Algorithm"] + metrics].copy()
    for m in metrics:
        df_melt[m] = pd.to_numeric(df_melt[m], errors="coerce")
    df_melt = df_melt.melt(id_vars=["Algorithm"], var_name="Metric", value_name="Score").dropna(subset=["Score"])

    y_rng = _dynamic_y_range(df_melt["Score"])

    fig_bar = px.bar(
        df_melt,
        x="Metric",
        y="Score",
        color="Algorithm",
        barmode="group",
        height=520,
        text="Score",
    )

    fig_bar.update_traces(
        texttemplate="%{y:.3f}",
        textposition="outside",
        cliponaxis=False,
    )

    fig_bar.update_layout(
        yaxis_title="Score",
        yaxis=dict(range=y_rng),
        plot_bgcolor="white",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        margin=dict(t=40, b=60, l=40, r=20),
        legend_title_text="Algorithm",
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------
    # F1-Per Class Comparison
    # -----------------------------
    st.subheader("F1-Per Class Comparison")

    existing = [c for c in PER_CLASS_COLS if c in df_chart.columns]
    if existing:
        f1df = df_chart[["Algorithm"] + existing].copy()
        for c in existing:
            f1df[c] = pd.to_numeric(f1df[c], errors="coerce")

        melted = f1df.melt(id_vars=["Algorithm"], var_name="Behaviour", value_name="F1").dropna(subset=["F1"])
        melted["Behaviour"] = melted["Behaviour"].str.extract(r"F1 \((.*)\)").fillna(melted["Behaviour"])

        behaviour_order = ["eating", "ruminating", "standing", "walking"]
        y_rng2 = _dynamic_y_range(melted["F1"])

        fig_f1 = px.bar(
            melted,
            x="Behaviour",
            y="F1",
            color="Algorithm",
            barmode="group",
            height=520,
            text="F1",
            category_orders={"Behaviour": behaviour_order},
        )
        fig_f1.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
        fig_f1.update_layout(
            xaxis_title="Behaviour",
            yaxis_title="F1-Score",
            yaxis=dict(range=y_rng2),
            plot_bgcolor="white",
            uniformtext_minsize=10,
            uniformtext_mode="hide",
            margin=dict(t=40, b=60, l=40, r=20),
            legend_title_text="Algorithm",
        )
        st.plotly_chart(fig_f1, use_container_width=True)
    else:
        st.info("Per-class F1 columns not found yet. Train at least one model that outputs per-class F1.")

# ======================================================
# TAB 2: Analysis (Energy/Latency)
# ======================================================
with tab2:
    st.subheader("Deployment Analysis")
    if "Target Device" in df_details.columns and not df_details.empty:
        # show most common / first
        td = df_details["Target Device"].dropna().astype(str)
        if not td.empty:
            st.caption(f"Hardware: **{td.iloc[0]}**")

    c1, c2 = st.columns(2)

    # For energy/latency, use df_chart (aggregated) to avoid repeated bars
    with c1:
        if "Energy (mJ)" in df_chart.columns:
            tmp = df_chart[["Algorithm", "Energy (mJ)"]].copy()
            tmp["Energy (mJ)"] = pd.to_numeric(tmp["Energy (mJ)"], errors="coerce")
            tmp = tmp.dropna(subset=["Energy (mJ)"])
            if not tmp.empty:
                fig_nrg = px.bar(tmp, x="Algorithm", y="Energy (mJ)", color="Algorithm", text="Energy (mJ)")
                fig_nrg.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
                st.plotly_chart(fig_nrg, use_container_width=True)
            else:
                st.info("Energy (mJ) values are missing/NaN.")
        else:
            st.info("Energy (mJ) is not available in results.")

    with c2:
        if "Latency (ms)" in df_chart.columns:
            tmp = df_chart[["Algorithm", "Latency (ms)"]].copy()
            tmp["Latency (ms)"] = pd.to_numeric(tmp["Latency (ms)"], errors="coerce")
            tmp = tmp.dropna(subset=["Latency (ms)"])
            if not tmp.empty:
                fig_lat = px.bar(tmp, x="Algorithm", y="Latency (ms)", color="Algorithm", text="Latency (ms)")
                fig_lat.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
                st.plotly_chart(fig_lat, use_container_width=True)
            else:
                st.info("Latency (ms) values are missing/NaN.")
        else:
            st.info("Latency (ms) is not available in results.")

# ======================================================
# TAB 3: Details (table)
# ======================================================
with tab3:
    st.subheader("All Results")

    # Full LOCO requirement: table shows cow folds + average, excludes final
    if view_mode == "Full LOCO (cow-wise average)":
        st.caption("Showing LOCO folds (cow1..cowN) + an additional 'average' row")
        st.dataframe(df_details, use_container_width=True)
    else:
        st.dataframe(df_details, use_container_width=True)

