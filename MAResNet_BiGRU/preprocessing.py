from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

STAGE_MAP: Dict[str, int] = {"Awake": 0, "Light": 1, "Deep": 2, "REM": 3}
STAGE_NAMES: List[str] = ["Awake", "Light", "Deep", "REM"]

BASE_FEATURE_COLS: List[str] = [
    "HR", "RMSSD", "HR_Stability", "SpO2", "Resp_Rate",
    "Apnea_Event", "SVM", "Body_Temp", "Ambient_Temp", "Humidity",
    "Light_Lux", "GSR_uS", "Audio_dB",
]

DELTA_SOURCES: List[str] = [
    "HR", "RMSSD", "SpO2", "Resp_Rate", "SVM", "Body_Temp", "GSR_uS", "Audio_dB",
]

ROLLING_SPECS = [
    ("HR", True),
    ("RMSSD", True),
    ("SpO2", False),
    ("GSR_uS", False),
    ("Audio_dB", False),
]

DELTA_COLS: List[str] = [f"{col}_delta" for col in DELTA_SOURCES]
ROLLING_COLS: List[str] = []
for _col, _with_std in ROLLING_SPECS:
    ROLLING_COLS.append(f"{_col}_roll5_mean")
    if _with_std:
        ROLLING_COLS.append(f"{_col}_roll5_std")

FEATURE_COLS: List[str] = BASE_FEATURE_COLS + DELTA_COLS + ROLLING_COLS


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    required = {"User_ID", "Day", "Timestamp", "Stage", *BASE_FEATURE_COLS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df.sort_values(["User_ID", "Day", "Timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Label"] = df["Stage"].map(STAGE_MAP)
    if df["Label"].isna().any():
        bad = sorted(df.loc[df["Label"].isna(), "Stage"].dropna().unique().tolist())
        raise ValueError(f"Unknown sleep stages found: {bad}")
    df["Label"] = df["Label"].astype(np.int8)

    for col in BASE_FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[BASE_FEATURE_COLS] = df[BASE_FEATURE_COLS].ffill().bfill().fillna(0.0)
    return add_temporal_features(df)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DELTA_SOURCES:
        df[f"{col}_delta"] = 0.0
    for col, with_std in ROLLING_SPECS:
        df[f"{col}_roll5_mean"] = 0.0
        if with_std:
            df[f"{col}_roll5_std"] = 0.0

    for _, group in df.groupby(["User_ID", "Day"], sort=False):
        idx = group.index
        for col in DELTA_SOURCES:
            df.loc[idx, f"{col}_delta"] = group[col].diff().fillna(0.0).to_numpy()
        for col, with_std in ROLLING_SPECS:
            rolling = group[col].rolling(5, min_periods=1)
            df.loc[idx, f"{col}_roll5_mean"] = rolling.mean().to_numpy()
            if with_std:
                df.loc[idx, f"{col}_roll5_std"] = rolling.std().fillna(0.0).to_numpy()
    return df


def user_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = np.array(df["User_ID"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    if len(users) < 3:
        train_df, tmp_df = _row_split(df, 1.0 - val_ratio - test_ratio, seed)
        val_share = val_ratio / max(val_ratio + test_ratio, 1e-8)
        val_df, test_df = _row_split(tmp_df, val_share, seed + 1)
        return train_df, val_df, test_df

    n_test = max(1, int(round(len(users) * test_ratio)))
    n_val = max(1, int(round(len(users) * val_ratio)))
    if n_test + n_val >= len(users):
        n_test, n_val = 1, 1

    test_users = set(users[:n_test])
    val_users = set(users[n_test:n_test + n_val])
    train_users = set(users[n_test + n_val:])
    return (
        df[df["User_ID"].isin(train_users)].copy(),
        df[df["User_ID"].isin(val_users)].copy(),
        df[df["User_ID"].isin(test_users)].copy(),
    )


def _row_split(df: pd.DataFrame, first_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = max(1, int(len(df) * first_ratio))
    return df.iloc[idx[:cut]].sort_index().copy(), df.iloc[idx[cut:]].sort_index().copy()


def fit_scaler(df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].to_numpy(dtype=np.float32))
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].to_numpy(dtype=np.float32))
    return df


def _window_group(group: pd.DataFrame, window_size: int, stride: int, feature_cols: List[str]):
    feats = group[feature_cols].to_numpy(dtype=np.float32)
    labels = group["Label"].to_numpy(dtype=np.int8)
    if len(group) < window_size:
        return (
            np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.int8),
        )

    center = window_size // 2
    xs, ys = [], []
    for start in range(0, len(group) - window_size + 1, stride):
        end = start + window_size
        xs.append(feats[start:end])
        ys.append(labels[start + center])
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int8)


def create_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 5,
    feature_cols: List[str] = FEATURE_COLS,
    n_jobs: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = [g for _, g in df.groupby(["User_ID", "Day"], sort=False)]
    runner = (
        delayed(_window_group)(g, window_size, stride, feature_cols)
        for g in groups
    )
    results = Parallel(n_jobs=n_jobs, backend="loky")(runner) if n_jobs != 1 else [
        _window_group(g, window_size, stride, feature_cols) for g in groups
    ]
    xs = [x for x, _ in results if len(x)]
    ys = [y for _, y in results if len(y)]
    if not xs:
        raise ValueError("No windows were created. Reduce window_size or check session lengths.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def oversample_minority_windows(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 0.75,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max() * target_ratio)
    rng = np.random.default_rng(seed)
    extra_x, extra_y = [], []
    for cls, count in zip(classes, counts):
        if count >= target:
            continue
        cls_idx = np.where(y == cls)[0]
        take = rng.choice(cls_idx, size=target - count, replace=True)
        extra_x.append(X[take])
        extra_y.append(y[take])
    if extra_x:
        X = np.concatenate([X, *extra_x], axis=0)
        y = np.concatenate([y, *extra_y], axis=0)
        order = rng.permutation(len(y))
        X, y = X[order], y[order]
    return X, y


def get_class_weights(y: np.ndarray, boost_minority: float = 1.35) -> Dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    out: Dict[int, float] = {}
    for cls, weight in zip(classes, weights):
        out[int(cls)] = float(weight * boost_minority if weight > 1.0 else weight)
    return out


def full_pipeline(
    dataset_path: str,
    window_size: int = 30,
    stride: int = 5,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    feature_cols: List[str] = FEATURE_COLS,
    seed: int = 42,
    n_jobs: int = 4,
) -> dict:
    raw = load_raw(dataset_path)
    train_df, val_df, test_df = user_split(raw, val_ratio, test_ratio, seed)
    scaler = fit_scaler(train_df, feature_cols)
    train_df = apply_scaler(train_df, scaler, feature_cols)
    val_df = apply_scaler(val_df, scaler, feature_cols)
    test_df = apply_scaler(test_df, scaler, feature_cols)

    X_train, y_train = create_windows(train_df, window_size, stride, feature_cols, n_jobs)
    X_val, y_val = create_windows(val_df, window_size, stride, feature_cols, n_jobs)
    X_test, y_test = create_windows(test_df, window_size, stride, feature_cols, n_jobs)
    X_train, y_train = oversample_minority_windows(X_train, y_train, seed=seed)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "class_weights": get_class_weights(y_train),
        "feature_cols": feature_cols,
        "stage_names": STAGE_NAMES,
    }

