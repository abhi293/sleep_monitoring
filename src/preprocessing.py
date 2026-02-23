"""
preprocessing.py
────────────────────────────────────────────────────────────────
Handles:
  • Raw CSV loading & type coercion
  • Per-user / per-session robust scaler fitting
  • Sliding-window segmentation using all 4 CPU cores (joblib)
  • Class-imbalance weight computation
  • Train / validation / test splitting by User_ID
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────

STAGE_MAP: Dict[str, int] = {"Awake": 0, "Light": 1, "Deep": 2, "REM": 3}
STAGE_NAMES: List[str] = ["Awake", "Light", "Deep", "REM"]

# Base sensor columns (from CSV)
BASE_FEATURE_COLS: List[str] = [
    "HR", "RMSSD", "HR_Stability", "SpO2", "Resp_Rate",
    "Apnea_Event", "SVM", "Body_Temp", "Ambient_Temp", "Humidity", "Light_Lux",
]

# Engineered temporal columns (added during loading)
DELTA_COLS: List[str] = [
    "HR_delta", "RMSSD_delta", "SpO2_delta",
    "Resp_Rate_delta", "SVM_delta", "Body_Temp_delta",
]
ROLLING_COLS: List[str] = [
    "HR_roll5_mean", "HR_roll5_std",
    "RMSSD_roll5_mean", "RMSSD_roll5_std",
    "SpO2_roll5_mean",
]

# Full feature set used by the model
FEATURE_COLS: List[str] = BASE_FEATURE_COLS + DELTA_COLS + ROLLING_COLS


# ────────────────────────────────────────────────────────────────
# Signal helpers
# ────────────────────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, lowcut: float, highcut: float,
              fs: float = 1.0, order: int = 3) -> np.ndarray:
    """Butterworth bandpass filter; falls back to raw if signal is too short."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if len(signal) < max(6, (order * 3)):
        return signal
    try:
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def _smooth(signal: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Simple moving-average smoothing."""
    if kernel < 2:
        return signal
    kernel = min(kernel, len(signal))
    return np.convolve(signal, np.ones(kernel) / kernel, mode="same")


# ────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-user-day delta and rolling features for richer temporal context."""
    delta_sources = ["HR", "RMSSD", "SpO2", "Resp_Rate", "SVM", "Body_Temp"]
    roll_sources  = [("HR", True), ("RMSSD", True), ("SpO2", False)]  # (col, add_std)

    # Pre-create columns with 0.0 to avoid fragmentation
    for col in delta_sources:
        df[f"{col}_delta"] = 0.0
    for col, add_std in roll_sources:
        df[f"{col}_roll5_mean"] = 0.0
        if add_std:
            df[f"{col}_roll5_std"] = 0.0

    for _, grp in df.groupby(["User_ID", "Day"], sort=False):
        idx = grp.index
        for col in delta_sources:
            df.loc[idx, f"{col}_delta"] = grp[col].diff().fillna(0.0).values
        for col, add_std in roll_sources:
            rm = grp[col].rolling(5, min_periods=1)
            df.loc[idx, f"{col}_roll5_mean"] = rm.mean().values
            if add_std:
                df.loc[idx, f"{col}_roll5_std"] = rm.std().fillna(0.0).values
    return df


def load_raw(path: str) -> pd.DataFrame:
    """Load CSV, coerce types, encode stage label, add temporal features."""
    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df.sort_values(["User_ID", "Day", "Timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Encode sleep stage
    df["Label"] = df["Stage"].map(STAGE_MAP).astype(np.int8)

    # Light smoothing on noisy channels (leave physiological channels intact)
    for col in ["Ambient_Temp", "Humidity", "Light_Lux"]:
        if col in df.columns:
            df[col] = _smooth(df[col].values, kernel=3)

    # Temporal feature engineering — deltas & rolling stats
    df = _add_temporal_features(df)

    logger.info("Dataset loaded: %d rows × %d cols | stages: %s",
                *df.shape, df["Stage"].value_counts().to_dict())
    return df


# ────────────────────────────────────────────────────────────────
# Scaler helpers
# ────────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLS
               ) -> RobustScaler:
    """Fit a global RobustScaler on the training split features."""
    scaler = RobustScaler()
    scaler.fit(df[feature_cols].values)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: RobustScaler,
                 feature_cols: List[str] = FEATURE_COLS) -> pd.DataFrame:
    """Return df with scaled feature columns."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df


# ────────────────────────────────────────────────────────────────
# Windowing (parallelized over users)
# ────────────────────────────────────────────────────────────────

def _window_user_day(
    group: pd.DataFrame,
    window_size: int,
    stride: int,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a single user-day group into overlapping windows.
    Uses CENTER-OF-WINDOW label instead of majority vote.
    This preserves minority stages (REM/Deep) that get outvoted
    by the dominant Light class at transition boundaries.
    Returns (X [N, W, F], y [N]) or empty arrays.
    """
    feats = group[feature_cols].values.astype(np.float32)   # [T, F]
    labels = group["Label"].values.astype(np.int8)          # [T]

    T = len(feats)
    if T < window_size:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), \
               np.empty(0, dtype=np.int8)

    center_offset = window_size // 2
    windows_X, windows_y = [], []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows_X.append(feats[start:end])
        # Center-of-window label — preserves minority stages
        windows_y.append(int(labels[start + center_offset]))

    X = np.stack(windows_X, axis=0)   # [N, W, F]
    y = np.array(windows_y, dtype=np.int8)
    return X, y


def create_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 10,
    feature_cols: List[str] = FEATURE_COLS,
    n_jobs: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window segmentation across all user-day sessions.
    Parallelized over groups using all CPU cores.
    Returns:
        X  → [N_total, window_size, n_features]
        y  → [N_total]
    """
    groups = [g for _, g in df.groupby(["User_ID", "Day"], sort=False)]
    logger.info("Windowing %d user-day groups (window=%d, stride=%d) on %d workers …",
                len(groups), window_size, stride, n_jobs)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_window_user_day)(g, window_size, stride, feature_cols)
        for g in groups
    )

    Xs = [r[0] for r in results if r[0].shape[0] > 0]
    ys = [r[1] for r in results if r[1].shape[0] > 0]

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    logger.info("Windows created: X=%s  y=%s", X_all.shape, y_all.shape)
    return X_all, y_all


# ────────────────────────────────────────────────────────────────
# Train / Val / Test split by User_ID
# ────────────────────────────────────────────────────────────────

def user_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Group-aware split: whole users go into train/val/test.
    Prevents data leakage across windows.
    """
    users = df["User_ID"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    n = len(users)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))

    test_users  = set(users[:n_test])
    val_users   = set(users[n_test:n_test + n_val])
    train_users = set(users[n_test + n_val:])

    df_train = df[df["User_ID"].isin(train_users)].copy()
    df_val   = df[df["User_ID"].isin(val_users)].copy()
    df_test  = df[df["User_ID"].isin(test_users)].copy()

    logger.info("Split → train %d users (%d rows) | val %d users (%d rows) | test %d users (%d rows)",
                len(train_users), len(df_train),
                len(val_users),   len(df_val),
                len(test_users),  len(df_test))
    return df_train, df_val, df_test


# ────────────────────────────────────────────────────────────────
# Class weights
# ────────────────────────────────────────────────────────────────

def get_class_weights(y: np.ndarray, boost_minority: float = 1.5) -> Dict[int, float]:
    """Compute balanced class weights with extra boost for minority stages.

    The standard 'balanced' weights are often insufficient for highly
    skewed distributions. We apply an additional multiplier to classes
    whose weight exceeds 1.0 (i.e., under-represented classes).
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = {}
    for c, w in zip(classes, weights):
        # Boost minority classes (weight > 1 means under-represented)
        if w > 1.0:
            w *= boost_minority
        cw[int(c)] = float(w)
    return cw


def oversample_minority_windows(
    X: np.ndarray, y: np.ndarray, target_ratio: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Duplicate minority-class windows to reduce class skew.

    target_ratio: minority classes are oversampled until they reach
                  this fraction of the majority class count.
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    target = int(max_count * target_ratio)

    extra_X, extra_y = [], []
    for cls, cnt in zip(classes, counts):
        if cnt < target:
            cls_mask = y == cls
            cls_X = X[cls_mask]
            cls_y = y[cls_mask]
            n_needed = target - cnt
            rng = np.random.default_rng(42)
            idxs = rng.choice(cnt, size=n_needed, replace=True)
            extra_X.append(cls_X[idxs])
            extra_y.append(cls_y[idxs])

    if extra_X:
        X = np.concatenate([X] + extra_X, axis=0)
        y = np.concatenate([y] + extra_y, axis=0)
        # Shuffle
        perm = np.random.default_rng(42).permutation(len(X))
        X, y = X[perm], y[perm]
        logger.info("Oversampled minorities → X=%s  y distribution: %s",
                    X.shape, dict(zip(*np.unique(y, return_counts=True))))
    return X, y


# ────────────────────────────────────────────────────────────────
# Convenience: full pipeline
# ────────────────────────────────────────────────────────────────

def full_pipeline(
    dataset_path: str,
    window_size: int = 30,
    stride: int = 10,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    feature_cols: List[str] = FEATURE_COLS,
    seed: int = 42,
    n_jobs: int = 4,
) -> dict:
    """
    End-to-end preprocessing: load → split by user → scale → window.
    Returns a dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                               scaler, class_weights, label_encoder.
    """
    df = load_raw(dataset_path)
    df_train, df_val, df_test = user_split(df, val_ratio, test_ratio, seed)

    # Fit scaler only on train
    scaler = fit_scaler(df_train, feature_cols)
    df_train = apply_scaler(df_train, scaler, feature_cols)
    df_val   = apply_scaler(df_val,   scaler, feature_cols)
    df_test  = apply_scaler(df_test,  scaler, feature_cols)

    X_train, y_train = create_windows(df_train, window_size, stride, feature_cols, n_jobs)
    X_val,   y_val   = create_windows(df_val,   window_size, stride, feature_cols, n_jobs)
    X_test,  y_test  = create_windows(df_test,  window_size, stride, feature_cols, n_jobs)

    # Oversample minority classes in training set to reduce skew
    X_train, y_train = oversample_minority_windows(X_train, y_train, target_ratio=0.6)

    cw = get_class_weights(y_train, boost_minority=1.5)
    logger.info("Class weights: %s", cw)

    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        scaler=scaler,
        class_weights=cw,
        feature_cols=feature_cols,
        stage_names=STAGE_NAMES,
    )
