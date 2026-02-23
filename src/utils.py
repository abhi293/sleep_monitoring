"""
utils.py
────────────────────────────────────────────────────────────────
Shared utilities:
  • Logging setup
  • Checkpoint / model persistence helpers
  • Metrics computation (accuracy, per-class, Cohen's kappa,
    sensitivity / false-alarm, HRV recovery proxy)
  • Multi-core TF configuration
  • Plotting helpers (confusion matrix, training curves,
    Pareto front, sleep hypnogram)
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, cohen_kappa_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)

STAGE_NAMES = ["Awake", "Light", "Deep", "REM"]
STAGE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]


# ────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Configure root logger with console + rotating file handler."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File
    fh = logging.FileHandler(Path(log_dir) / "sleep_monitor.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ────────────────────────────────────────────────────────────────
# Multi-core TF configuration
# ────────────────────────────────────────────────────────────────

def configure_tf_cpu(n_cores: int = 4) -> None:
    """
    Configure TensorFlow to use all available CPU cores.
    • intra_op  → parallelism within a single op (e.g., matrix mul threads)
    • inter_op  → parallelism across independent ops (graph-level)
    """
    tf.config.threading.set_intra_op_parallelism_threads(n_cores)
    tf.config.threading.set_inter_op_parallelism_threads(n_cores)

    # Enable oneDNN/MKL-DNN optimisations (uses AVX2 automatically on EPYC)
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
    # Use all logical CPUs
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_cores))
    logger.info("TF CPU parallelism configured: %d threads", n_cores)


# ────────────────────────────────────────────────────────────────
# Keras callbacks
# ────────────────────────────────────────────────────────────────

def get_callbacks(
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    patience: int = 8,
    lr_schedule: str = "cosine",
    epochs: int = 50,
    learning_rate: float = 1e-3,
) -> List[tf.keras.callbacks.Callback]:
    """Return a list of useful training callbacks."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save best val_accuracy checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(checkpoint_dir) / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # CSV training log
        tf.keras.callbacks.CSVLogger(
            str(Path(log_dir) / "training_log.csv"), append=False
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    if lr_schedule == "reduce_on_plateau":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
            )
        )
    elif lr_schedule == "cosine":
        steps_per_epoch_est = 1  # will be updated at train time
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: float(
                    learning_rate * 0.5 * (1 + np.cos(np.pi * epoch / max(epochs, 1)))
                ),
                verbose=0,
            )
        )
    return callbacks


# ────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
) -> Dict:
    """Full classification metrics + Cohen's kappa + per-class sensitivity/FAR."""
    report = classification_report(
        y_true, y_pred,
        target_names=stage_names,
        output_dict=True,
        zero_division=0,
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred)

    # Per-class false alarm rate = FP / (FP + TN)
    far_per_class = {}
    for i, name in enumerate(stage_names):
        tp  = cm[i, i]
        fp  = cm[:, i].sum() - tp
        fn  = cm[i, :].sum() - tp
        tn  = cm.sum() - tp - fp - fn
        far = fp / max(fp + tn, 1)
        far_per_class[name] = float(far)

    # Sleep efficiency: fraction of windows classified as sleep (not Awake)
    sleep_efficiency = float((y_pred > 0).sum() / len(y_pred))

    return {
        "accuracy": report["accuracy"],
        "kappa":    kappa,
        "per_class": report,
        "false_alarm_rate_per_class": far_per_class,
        "mean_false_alarm_rate": float(np.mean(list(far_per_class.values()))),
        "sleep_efficiency": sleep_efficiency,
        "confusion_matrix": cm.tolist(),
    }


def hrv_recovery_score(rmssd_sequence: np.ndarray) -> float:
    """
    Simple HRV recovery proxy: slope of RMSSD over the night.
    Positive slope → improving recovery; negative → deteriorating.
    Returns normalised score ∈ [0, 1].
    """
    if len(rmssd_sequence) < 2:
        return 0.5
    x   = np.arange(len(rmssd_sequence))
    slope, _ = np.polyfit(x, rmssd_sequence, 1)
    # Normalise: map [−max_abs, +max_abs] → [0, 1]
    max_abs = max(abs(slope), 1e-6)
    return float(np.clip(0.5 + slope / (2 * max_abs), 0.0, 1.0))


def sleep_quality_score(
    accuracy: float,
    kappa: float,
    mean_far: float,
    sleep_efficiency: float,
) -> float:
    """Composite sleep quality score ∈ [0, 100]."""
    score = (
        0.40 * accuracy
        + 0.25 * max(kappa, 0.0)
        + 0.20 * sleep_efficiency
        + 0.15 * (1.0 - mean_far)
    )
    return round(score * 100, 2)


# ────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────

def save_scaler(scaler, path: str = "checkpoints/scaler.pkl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved → %s", path)


def load_scaler(path: str = "checkpoints/scaler.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(metrics: dict, path: str = "logs/metrics.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy arrays to lists for JSON serialisation
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(metrics), f, indent=2)
    logger.info("Metrics saved → %s", path)


# ────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────

def plot_training_history(
    history: tf.keras.callbacks.History,
    save_path: str = "logs/training_curves.png",
) -> None:
    """Plot loss and accuracy curves side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy")],
        ["Loss", "Accuracy"],
    ):
        ax.plot(history.history[metric[0]], label="Train", linewidth=2)
        ax.plot(history.history[metric[1]], label="Val",   linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.legend(); ax.grid(True, alpha=0.4)

    plt.suptitle("SleepNet — Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training curves saved → %s", save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/confusion_matrix.png",
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=stage_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title("Sleep Stage Classification — Normalised Confusion Matrix",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved → %s", save_path)


def plot_hypnogram(
    y_pred: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/hypnogram.png",
    title: str = "Predicted Sleep Hypnogram",
) -> None:
    """Plot a colour-coded hypnogram from predicted labels."""
    fig, ax = plt.subplots(figsize=(16, 4))
    time_min = np.arange(len(y_pred))

    # Fill coloured bands
    for i in range(len(y_pred) - 1):
        ax.fill_between(
            [time_min[i], time_min[i + 1]],
            [0, 0], [1, 1],
            color=STAGE_COLORS[y_pred[i]],
            alpha=0.7,
        )

    ax.set_yticks([])
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    patches = [mpatches.Patch(color=STAGE_COLORS[i], label=stage_names[i])
               for i in range(len(stage_names))]
    ax.legend(handles=patches, loc="upper right", ncol=4)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Hypnogram saved → %s", save_path)


def plot_pareto_front(
    pareto_archive: List[dict],
    save_path: str = "mopso_results/pareto_front.png",
) -> None:
    """2-D scatter of Pareto archive objectives (accuracy vs false alarm rate)."""
    if not pareto_archive:
        return
    objs = np.array([s["objectives"] for s in pareto_archive])
    # objectives: [1-acc, false_alarm, log_params]
    acc_vals  = 1 - objs[:, 0]
    far_vals  = objs[:, 1]
    size_vals = objs[:, 2]

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(acc_vals, far_vals, c=size_vals, cmap="viridis",
                    s=80, zorder=3, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="log₁₀(param count)")
    ax.set_xlabel("Validation Accuracy", fontsize=12)
    ax.set_ylabel("False Alarm Rate", fontsize=12)
    ax.set_title("MOPSO Pareto Front — Accuracy vs False Alarm Rate",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Pareto front plot saved → %s", save_path)
