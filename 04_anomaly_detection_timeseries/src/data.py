"""
Data loading and preprocessing for time series anomaly detection.

Supports:
- Synthetic time series with configurable anomalies
- CSV upload (expects value column, optional timestamp and anomaly label)
- Sliding window feature extraction for Isolation Forest and VAE
"""

import io
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeSeriesData:
    """Container for loaded time series with optional ground-truth labels."""
    values: np.ndarray
    labels: Optional[np.ndarray] = None  # 0 = normal, 1 = anomaly (if available)
    timestamps: Optional[np.ndarray] = None
    name: str = "series"

    @property
    def n_points(self) -> int:
        return len(self.values)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"value": self.values})
        if self.timestamps is not None:
            df["timestamp"] = self.timestamps
        if self.labels is not None:
            df["anomaly"] = self.labels.astype(int)
        return df


def generate_synthetic(
    n_points: int = 2000,
    anomaly_ratio: float = 0.02,
    seed: Optional[int] = None,
) -> TimeSeriesData:
    """
    Generate synthetic time series with realistic anomalies.

    Uses a sinusoidal baseline + noise. Anomalies are:
    - Spikes (sudden peaks)
    - Level shifts (baseline jumps)
    - Drift (gradual drift away from baseline)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_points)
    baseline = 10 * np.sin(t) + 0.5 * np.sin(5 * t)
    noise = rng.normal(0, 0.5, n_points)
    values = baseline + noise

    labels = np.zeros(n_points, dtype=int)
    n_anomalies = int(n_points * anomaly_ratio)
    anomaly_indices = rng.choice(n_points, size=n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = rng.choice(["spike", "level", "drift"])
        if anomaly_type == "spike":
            values[idx] += rng.uniform(5, 12) * rng.choice([-1, 1])
        elif anomaly_type == "level":
            shift = rng.uniform(3, 8) * rng.choice([-1, 1])
            window = min(20, n_points - idx)
            values[idx : idx + window] += shift
            labels[idx : idx + window] = 1
            continue
        else:  # drift
            drift_len = min(50, n_points - idx)
            drift = np.linspace(0, rng.uniform(3, 7), drift_len) * rng.choice([-1, 1])
            values[idx : idx + drift_len] += drift
            labels[idx : idx + drift_len] = 1
            continue
        labels[idx] = 1

    return TimeSeriesData(
        values=values.astype(np.float32),
        labels=labels,
        timestamps=np.arange(n_points, dtype=float),
        name="synthetic",
    )


def load_csv(
    file_content: bytes | str,
    value_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> TimeSeriesData:
    """
    Load time series from CSV.

    If columns are not specified, infers:
    - value: first numeric column, or column named 'value', 'values', 'y'
    - timestamp: first datetime-like or 'timestamp', 'time', 't', 'x'
    - label: column named 'anomaly', 'label', 'is_anomaly' (0/1)
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")
    df = pd.read_csv(io.StringIO(file_content))

    # Infer value column
    if value_column is None:
        candidates = ["value", "values", "y", "data"]
        for c in candidates:
            if c in df.columns:
                value_column = c
                break
        if value_column is None:
            numeric = df.select_dtypes(include=[np.number]).columns
            value_column = numeric[0] if len(numeric) > 0 else df.columns[0]

    values = df[value_column].values.astype(np.float32)
    values = np.nan_to_num(values, nan=np.nanmean(values))

    # Timestamps
    timestamps = None
    if timestamp_column and timestamp_column in df.columns:
        ts = pd.to_datetime(df[timestamp_column], errors="coerce")
        timestamps = ts.astype(np.int64).values if ts.notna().all() else np.arange(len(values))
    else:
        for c in ["timestamp", "time", "t", "x", "index"]:
            if c in df.columns:
                try:
                    timestamps = pd.to_numeric(df[c]).values
                    break
                except Exception:
                    pass
        if timestamps is None:
            timestamps = np.arange(len(values), dtype=float)

    # Labels
    labels = None
    if label_column and label_column in df.columns:
        labels = (df[label_column].values != 0).astype(int)
    else:
        for c in ["anomaly", "label", "is_anomaly"]:
            if c in df.columns:
                labels = (df[c].values != 0).astype(int)
                break

    return TimeSeriesData(
        values=values,
        labels=labels,
        timestamps=timestamps,
        name="uploaded",
    )


def sliding_windows(
    values: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Extract sliding windows from 1D time series.

    Returns shape (n_windows, window_size).
    """
    n = len(values)
    if n < window_size:
        raise ValueError(f"Series length {n} < window_size {window_size}")
    n_windows = (n - window_size) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_size
    )[::stride]
    return windows.astype(np.float32)


def normalize_windows(windows: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize each window for model input."""
    if method == "zscore":
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return (windows - mean) / std
    elif method == "minmax":
        min_ = windows.min(axis=1, keepdims=True)
        max_ = windows.max(axis=1, keepdims=True)
        span = np.where(max_ - min_ < 1e-8, 1.0, max_ - min_)
        return (windows - min_) / span
    return windows
