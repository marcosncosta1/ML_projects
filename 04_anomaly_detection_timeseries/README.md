# Time Series Anomaly Detection

Interactive anomaly detection on time series using **Isolation Forest** and **VAE** (Variational Autoencoder).

## Overview

- **Isolation Forest** — Uses sliding-window statistics (mean, std, quantiles) and isolates anomalies in fewer random splits. Fast, no training.
- **VAE** — Trains on normal windows; reconstruction error = anomaly score. Captures complex temporal patterns.

## Quick Start

```bash
cd 04_anomaly_detection_timeseries
pip install -r requirements.txt
streamlit run app.py
```

## Data Sources

1. **Synthetic (demo)** — Generate a sinusoidal time series with configurable spikes, level shifts, and drift anomalies.
2. **Upload CSV** — Your own data. Expects a value column (auto-detected: `value`, `y`, `data` or first numeric). Optional: `timestamp`, `anomaly` (0/1).

## Features

- Sliding window feature extraction
- Isolation Forest baseline
- VAE-based anomaly scoring
- Interactive Plotly charts: signal + color-coded anomaly markers
- Threshold slider (percentile) to adjust sensitivity
- Side-by-side comparison of both methods
- Method correlation analysis when both are run

## File Structure

```
04_anomaly_detection_timeseries/
├── app.py              # Streamlit dashboard
├── src/
│   ├── data.py         # Load CSV, synthetic data, sliding windows
│   ├── isolation_forest.py  # Isolation Forest scoring
│   └── vae.py          # VAE model and training
├── requirements.txt
├── README.md
└── PROJECT.md
```

## Datasets (optional)

The app works out-of-the-box with synthetic data. For real benchmarks:

- **Yahoo Webscope S5** — [Hugging Face](https://huggingface.co/datasets/YahooResearch/ydata-labeled-time-series-anomalies-v1_0)
- **NASA SMAP/MSL** — Industrial sensor anomalies
