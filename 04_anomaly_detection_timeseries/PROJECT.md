# Anomaly Detection on Time Series

## Overview
An interactive anomaly detection app applied to public time series data (ECG signals or industrial sensor data). Uses Isolation Forest and/or a Variational Autoencoder (VAE) to score and visualize anomalies.

## Category
Machine Learning / Unsupervised Learning / Industry Applications

## Stack
- **scikit-learn** — Isolation Forest
- **PyTorch** — VAE implementation
- **Plotly / Streamlit** — interactive visualization
- **pandas / numpy** — data processing

## Datasets
- **MIT-BIH ECG Arrhythmia** (via PhysioNet or Kaggle)
- **NASA SMAP/MSL** sensor anomaly dataset
- **Yahoo Webscope S5** (time series benchmarks)

## Key Features
- Load a time series dataset (or upload CSV)
- Run Isolation Forest and/or VAE-based anomaly scoring
- Interactive Plotly chart: signal + color-coded anomaly scores
- Threshold slider to adjust sensitivity
- Side-by-side comparison of both methods

## Architecture
### Isolation Forest
1. Sliding window features extracted from time series
2. Isolation Forest scores each window
3. Scores mapped back to timeline

### VAE
1. Train on "normal" subsequences
2. Reconstruction error on test windows = anomaly score
3. High reconstruction error → anomaly

## Portfolio Value
- Highly relevant to industry (manufacturing, healthcare, fintech)
- Demonstrates unsupervised ML beyond classification
- Interactive visualization makes results tangible

## Milestones
- [ ] Data loading + preprocessing pipeline
- [ ] Isolation Forest baseline
- [ ] VAE implementation + training
- [ ] Streamlit dashboard with interactive plots
- [ ] README with dataset description and method comparison

## Notes
<!-- Add implementation notes, decisions, and progress here -->
