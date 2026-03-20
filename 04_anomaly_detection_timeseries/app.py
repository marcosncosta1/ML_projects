"""
Anomaly Detection on Time Series — Streamlit app
=================================================

Interactive dashboard for:
- Isolation Forest anomaly scoring
- VAE-based anomaly scoring
- Side-by-side comparison with threshold slider
- CSV upload or synthetic data

Run:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from src.data import TimeSeriesData, generate_synthetic, load_csv, sliding_windows
from src.isolation_forest import fit_isolation_forest, score_timeline
from src.vae import train_vae, vae_anomaly_scores

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detection — Time Series",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar: data source ────────────────────────────────────────────────────
st.sidebar.header("📂 Data")
data_source = st.sidebar.radio(
    "Data source",
    ["Synthetic (demo)", "Upload CSV"],
    index=0,
)

data: TimeSeriesData | None = None

if data_source == "Synthetic (demo)":
    n_points = st.sidebar.slider("Series length", 500, 5000, 2000, 100)
    anomaly_ratio = st.sidebar.slider("Anomaly ratio", 0.01, 0.1, 0.02, 0.01)
    seed = st.sidebar.number_input("Random seed", 0, 99999, 42)
    if st.sidebar.button("Generate"):
        data = generate_synthetic(n_points=n_points, anomaly_ratio=anomaly_ratio, seed=seed)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        content = uploaded.read()
        try:
            data = load_csv(content)
        except Exception as e:
            st.sidebar.error(f"Failed to load CSV: {e}")
            data = None
    else:
        data = None

# Default: generate if no upload
if data is None and data_source == "Synthetic (demo)":
    data = generate_synthetic(n_points=2000, anomaly_ratio=0.02, seed=42)

# ── Parameters (must be defined before use) ───────────────────────────────────
st.sidebar.header("⚙️ Parameters")
window_size = st.sidebar.slider("Window size", 16, 128, 64, 8)
run_if = st.sidebar.checkbox("Run Isolation Forest", value=True)
run_vae = st.sidebar.checkbox("Run VAE", value=True)
vae_epochs = st.sidebar.slider("VAE epochs", 10, 100, 50, 10) if run_vae else 50

# ── Main content ─────────────────────────────────────────────────────────────
st.title("📈 Time Series Anomaly Detection")
st.markdown(
    "Detect anomalies using **Isolation Forest** (statistical) and **VAE** (reconstruction error). "
    "Adjust the threshold to control sensitivity."
)

if data is None:
    st.info("Upload a CSV file or use synthetic data to get started.")
    st.stop()

if not run_if and not run_vae:
    st.warning("Enable at least one method (Isolation Forest or VAE) in the sidebar.")
    st.stop()

values = data.values
n = len(values)

# ── Run models ───────────────────────────────────────────────────────────────
@st.cache_data
def run_isolation_forest(_values: np.ndarray, window: int):
    model, _, _ = fit_isolation_forest(_values, window_size=window)
    scores = score_timeline(model, _values, window_size=window)
    return scores


@st.cache_data
def run_vae_scoring(_values: np.ndarray, window: int, epochs: int):
    windows = sliding_windows(_values, window)
    model, _ = train_vae(windows, epochs=epochs, verbose=False)
    scores = vae_anomaly_scores(model, _values, window_size=window)
    return scores


with st.spinner("Running anomaly detection…"):
    iforest_scores = run_isolation_forest(values, window_size) if run_if else None
    vae_scores = run_vae_scoring(values, window_size, vae_epochs) if run_vae else None

# ── Threshold slider ─────────────────────────────────────────────────────────
st.sidebar.header("🎚️ Threshold")
threshold_pct = st.sidebar.slider(
    "Anomaly threshold (percentile)",
    90.0,
    99.9,
    95.0,
    0.5,
    help="Points above this percentile of scores are flagged as anomalies.",
)

# ── Plot ─────────────────────────────────────────────────────────────────────
x = np.arange(n) if data.timestamps is None else data.timestamps

# Collect scores and normalise
scores_list = []
labels_list = []
if run_if and iforest_scores is not None:
    scores_list.append(iforest_scores)
    labels_list.append("Isolation Forest")
if run_vae and vae_scores is not None:
    scores_list.append(vae_scores)
    labels_list.append("VAE")

norm_scores = []
for s in scores_list:
    s_min, s_max = s.min(), s.max()
    if s_max - s_min > 1e-8:
        ns = (s - s_min) / (s_max - s_min)
    else:
        ns = np.zeros_like(s)
    norm_scores.append(ns)

# Points above this (normalised) threshold are anomalies
threshold = (
    np.percentile(np.concatenate(norm_scores), threshold_pct)
    if norm_scores
    else 0.0
)

# Combined anomaly mask: flagged by any method
any_anomaly = np.any(np.column_stack([s >= threshold for s in norm_scores]), axis=1)

n_rows = 1 + len(scores_list)
subplot_titles = ["Time series"] + [f"{l} score" for l in labels_list]
fig = make_subplots(
    rows=n_rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=subplot_titles,
)

# Row 1: Signal + anomaly markers
fig.add_trace(
    go.Scatter(x=x, y=values, name="Signal", line=dict(color="#4A90D9", width=1.5)),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x[any_anomaly],
        y=values[any_anomaly],
        mode="markers",
        name="Anomalies",
        marker=dict(size=5, color="red", symbol="x", line=dict(width=1)),
    ),
    row=1,
    col=1,
)

# Rows 2+: Score curves + threshold
for i, (norm_s, label) in enumerate(zip(norm_scores, labels_list)):
    row = 2 + i
    fig.add_trace(
        go.Scatter(
            x=x,
            y=norm_s,
            name=label,
            line=dict(width=1.5),
        ),
        row=row,
        col=1,
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Threshold ({threshold_pct}%)",
        row=row,
        col=1,
    )

fig.update_layout(
    height=250 * n_rows,
    margin=dict(l=50, r=50, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified",
)
fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Value", row=1, col=1)
for i in range(len(scores_list)):
    fig.update_yaxes(title_text="Score (norm)", row=2 + i, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Metrics ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Data points", f"{n:,}")
with col2:
    if data.labels is not None:
        n_true = data.labels.sum()
        st.metric("True anomalies (if labeled)", f"{int(n_true)} ({100*n_true/n:.1f}%)")
with col3:
    if norm_scores:
        n_pred = (np.concatenate(norm_scores) >= threshold).sum()
        st.metric("Predicted anomalies", f"{int(n_pred)} ({100*n_pred/n:.1f}%)")

# ── Method comparison (if both run) ───────────────────────────────────────────
if run_if and run_vae and iforest_scores is not None and vae_scores is not None:
    st.subheader("Method comparison")
    df = pd.DataFrame({
        "Isolation Forest": iforest_scores,
        "VAE": vae_scores,
    })
    corr = np.corrcoef(iforest_scores, vae_scores)[0, 1]
    st.markdown(
        f"Correlation between methods: **{corr:.3f}**. "
        "High correlation suggests both agree on anomaly regions."
    )

# ── How it works ─────────────────────────────────────────────────────────────
with st.expander("How it works"):
    st.markdown(
        """
        ### Isolation Forest
        - Extracts sliding windows from the time series
        - Computes per-window statistics (mean, std, min, max, quantiles)
        - Isolation Forest isolates anomalies in fewer random splits → lower path length → higher score
        - Scores mapped back to the timeline

        ### VAE (Variational Autoencoder)
        - Trains on sliding windows (normalised)
        - Learns to reconstruct "normal" patterns
        - Anomalies have high reconstruction error → high score

        ### When to use which?
        | Method | Pros | Cons |
        |--------|------|------|
        | Isolation Forest | Fast, no training, interpretable | Uses summary stats, may miss subtle patterns |
        | VAE | Captures complex temporal patterns | Needs training, slower |
        """
    )
