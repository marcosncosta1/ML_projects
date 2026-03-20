"""
Isolation Forest-based anomaly scoring for time series.

Uses sliding window features (statistics per window) fed into Isolation Forest.
Scores are mapped back to the original timeline via the window centre.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional

from .data import sliding_windows, TimeSeriesData


def extract_window_features(windows: np.ndarray) -> np.ndarray:
    """
    Extract summary statistics from each window for Isolation Forest.

    Per window: mean, std, min, max, range, median, q25, q75.
    """
    feats = []
    feats.append(windows.mean(axis=1))
    feats.append(windows.std(axis=1))
    feats.append(windows.min(axis=1))
    feats.append(windows.max(axis=1))
    feats.append(windows.max(axis=1) - windows.min(axis=1))
    feats.append(np.median(windows, axis=1))
    feats.append(np.percentile(windows, 25, axis=1))
    feats.append(np.percentile(windows, 75, axis=1))
    feats = np.column_stack(feats)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def fit_isolation_forest(
    values: np.ndarray,
    window_size: int = 64,
    stride: int = 1,
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: Optional[int] = None,
) -> tuple[IsolationForest, np.ndarray, np.ndarray]:
    """
    Fit Isolation Forest on sliding-window features.

    Returns:
        model: fitted IsolationForest
        windows: raw windows (n_windows, window_size)
        features: extracted features (n_windows, n_features)
    """
    windows = sliding_windows(values, window_size, stride)
    features = extract_window_features(windows)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features)
    return model, windows, features


def score_timeline(
    model: IsolationForest,
    values: np.ndarray,
    window_size: int = 64,
    stride: int = 1,
) -> np.ndarray:
    """
    Compute anomaly scores for each time point.

    Uses the score from the window that contains each point.
    If multiple windows contain a point, average their (negated) decision function.
    Higher = more anomalous.
    """
    windows = sliding_windows(values, window_size, stride)
    features = extract_window_features(windows)
    # decision_function: negative = anomaly, positive = normal
    # We negate so higher = more anomalous (matches intuition)
    raw_scores = -model.decision_function(features)

    # Map window scores back to timeline
    n = len(values)
    scores = np.full(n, np.nan)
    counts = np.zeros(n)

    for i in range(len(windows)):
        start = i * stride
        end = start + window_size
        for j in range(start, min(end, n)):
            if np.isnan(scores[j]):
                scores[j] = raw_scores[i]
                counts[j] = 1
            else:
                scores[j] += raw_scores[i]
                counts[j] += 1

    counts = np.where(counts == 0, 1, counts)
    scores = scores / counts
    scores = np.nan_to_num(scores, nan=0.0)
    return scores.astype(np.float32)
