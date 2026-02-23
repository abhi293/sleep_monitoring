"""
utils.py
────────────────────────────────────────────────────────────────
Shared utilities:
  • Logging setup
  • Checkpoint / model persistence helpers
  • Metrics computation (accuracy, per-class, Cohen's kappa,
    sensitivity / false-alarm, HRV recovery proxy)
  • Multi-core TF configuration
  • Comprehensive plotting helpers (16+ plot types, high-DPI,
    bold fonts, vivid contrasty colour palette)
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
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, cohen_kappa_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)

logger = logging.getLogger(__name__)

STAGE_NAMES = ["Awake", "Light", "Deep", "REM"]
# High-contrast vivid palette
STAGE_COLORS = ["#E63946", "#1D7AF2", "#06D6A0", "#AB47BC"]

# ── Global matplotlib style for ALL plots ─────────────────────
_PLOT_STYLE: Dict = {
    "DPI": 350,
    "TITLE_SIZE": 22,
    "AXIS_LABEL_SIZE": 17,
    "TICK_SIZE": 14,
    "LEGEND_SIZE": 14,
    "ANNOT_SIZE": 13,
    "SUPTITLE_SIZE": 24,
    "FONT_WEIGHT": "bold",
}

plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.titlesize": _PLOT_STYLE["TITLE_SIZE"],
    "axes.titleweight": "bold",
    "axes.labelsize": _PLOT_STYLE["AXIS_LABEL_SIZE"],
    "axes.labelweight": "bold",
    "xtick.labelsize": _PLOT_STYLE["TICK_SIZE"],
    "ytick.labelsize": _PLOT_STYLE["TICK_SIZE"],
    "legend.fontsize": _PLOT_STYLE["LEGEND_SIZE"],
    "figure.dpi": _PLOT_STYLE["DPI"],
    "savefig.dpi": _PLOT_STYLE["DPI"],
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Vivid qualitative palette for multi-line / bar charts
_VIVID_PALETTE = ["#E63946", "#1D7AF2", "#06D6A0", "#AB47BC",
                  "#FF9F1C", "#2EC4B6", "#E71D73", "#011627"]


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
# Plotting — Comprehensive Suite (16+ plot types)
# ────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, save_path: str) -> None:
    """Helper: ensure parent dir exists, save at high DPI, close."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_PLOT_STYLE["DPI"], bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    logger.info("Plot saved → %s", save_path)


# ── 1. Training History (Loss + Accuracy + LR) ────────────────

def plot_training_history(
    history,
    save_path: str = "logs/training_curves.png",
) -> None:
    """Plot loss, accuracy, and learning rate curves."""
    h = history.history if hasattr(history, "history") else history
    has_lr = "lr" in h or "learning_rate" in h
    n_cols = 3 if has_lr else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

    # --- Loss ---
    ax = axes[0]
    ax.plot(h["loss"],     label="Train", linewidth=2.5, color=_VIVID_PALETTE[0])
    ax.plot(h["val_loss"], label="Val",   linewidth=2.5, color=_VIVID_PALETTE[1],
            linestyle="--")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(frameon=True, shadow=True)

    # --- Accuracy ---
    ax = axes[1]
    ax.plot(h["accuracy"],     label="Train", linewidth=2.5, color=_VIVID_PALETTE[2])
    ax.plot(h["val_accuracy"], label="Val",   linewidth=2.5, color=_VIVID_PALETTE[3],
            linestyle="--")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(frameon=True, shadow=True)

    # --- Learning Rate ---
    if has_lr:
        ax = axes[2]
        lr_key = "lr" if "lr" in h else "learning_rate"
        ax.plot(h[lr_key], linewidth=2.5, color=_VIVID_PALETTE[4])
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.suptitle("SleepNet — Training History",
                 fontsize=_PLOT_STYLE["SUPTITLE_SIZE"], fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 2. Confusion Matrix (normalised, vibrant) ─────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/confusion_matrix.png",
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))

    cmap = plt.cm.YlOrRd  # vivid orange-red gradient
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=_PLOT_STYLE["TICK_SIZE"])

    n = len(stage_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(stage_names, fontweight="bold")
    ax.set_yticklabels(stage_names, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalised Confusion Matrix")

    # Annotate cells with value
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            colour = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center", color=colour,
                    fontsize=_PLOT_STYLE["ANNOT_SIZE"] + 2, fontweight="bold")

    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 3. Confusion Matrix — Raw Counts ──────────────────────────

def plot_confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/confusion_matrix_counts.png",
) -> None:
    """Confusion matrix with raw sample counts."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))

    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=_PLOT_STYLE["TICK_SIZE"])

    n = len(stage_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(stage_names, fontweight="bold")
    ax.set_yticklabels(stage_names, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Raw Counts")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            colour = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", color=colour,
                    fontsize=_PLOT_STYLE["ANNOT_SIZE"], fontweight="bold")

    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 4. Hypnogram ──────────────────────────────────────────────

def plot_hypnogram(
    y_pred: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/hypnogram.png",
    title: str = "Predicted Sleep Hypnogram",
) -> None:
    """Colour-coded band hypnogram with a step-line overlay."""
    fig, ax = plt.subplots(figsize=(18, 5))
    t = np.arange(len(y_pred))

    # Colour bands
    for i in range(len(y_pred) - 1):
        ax.axvspan(t[i], t[i + 1], color=STAGE_COLORS[y_pred[i]], alpha=0.65)

    # Step-line overlay for readability
    ax.step(t, y_pred, where="post", color="#1a1a2e", linewidth=1.6, alpha=0.85)

    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names, fontweight="bold")
    ax.set_xlabel("Window Index (Time →)")
    ax.set_title(title)
    ax.set_xlim(0, len(y_pred) - 1)
    ax.set_ylim(-0.5, len(stage_names) - 0.5)
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=STAGE_COLORS[i], label=stage_names[i])
               for i in range(len(stage_names))]
    ax.legend(handles=patches, loc="upper right", ncol=4, frameon=True, shadow=True)

    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 5. True vs Predicted Hypnogram Comparison ─────────────────

def plot_hypnogram_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/hypnogram_comparison.png",
    max_windows: int = 600,
) -> None:
    """Side-by-side true vs predicted hypnogram."""
    n = min(max_windows, len(y_true))
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    for ax, y, label in zip(axes, [y_true[:n], y_pred[:n]], ["Ground Truth", "Predicted"]):
        t = np.arange(len(y))
        for i in range(len(y) - 1):
            ax.axvspan(t[i], t[i + 1], color=STAGE_COLORS[y[i]], alpha=0.65)
        ax.step(t, y, where="post", color="#1a1a2e", linewidth=1.4, alpha=0.85)
        ax.set_yticks(range(len(stage_names)))
        ax.set_yticklabels(stage_names, fontweight="bold")
        ax.set_ylim(-0.5, len(stage_names) - 0.5)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=_PLOT_STYLE["TITLE_SIZE"] - 2)
        patches = [mpatches.Patch(color=STAGE_COLORS[i], label=stage_names[i])
                   for i in range(len(stage_names))]
        ax.legend(handles=patches, loc="upper right", ncol=4, frameon=True)

    axes[-1].set_xlabel("Window Index (Time →)")
    fig.suptitle("Hypnogram — True vs Predicted",
                 fontsize=_PLOT_STYLE["SUPTITLE_SIZE"], fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 6. Pareto Front ──────────────────────────────────────────

def plot_pareto_front(
    pareto_archive: List[dict],
    save_path: str = "mopso_results/pareto_front.png",
) -> None:
    """2-D scatter of Pareto archive objectives."""
    if not pareto_archive:
        return
    objs = np.array([s["objectives"] for s in pareto_archive])
    acc_vals  = 1 - objs[:, 0]
    far_vals  = objs[:, 1]
    size_vals = objs[:, 2]

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(acc_vals, far_vals, c=size_vals, cmap="plasma",
                    s=120, zorder=3, edgecolors="black", linewidths=1.0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log₁₀(param count)", fontsize=_PLOT_STYLE["AXIS_LABEL_SIZE"])
    cbar.ax.tick_params(labelsize=_PLOT_STYLE["TICK_SIZE"])
    ax.set_xlabel("Validation Accuracy")
    ax.set_ylabel("False Alarm Rate")
    ax.set_title("MOPSO Pareto Front — Accuracy vs FAR")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 7. Per-Class Metrics Bar Chart ────────────────────────────

def plot_per_class_metrics(
    metrics: Dict,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/per_class_metrics.png",
) -> None:
    """Grouped bar chart: Precision, Recall, F1-Score per stage."""
    prec = [metrics["per_class"].get(s, {}).get("precision", 0) for s in stage_names]
    rec  = [metrics["per_class"].get(s, {}).get("recall", 0)    for s in stage_names]
    f1   = [metrics["per_class"].get(s, {}).get("f1-score", 0)  for s in stage_names]

    x = np.arange(len(stage_names))
    w = 0.22

    fig, ax = plt.subplots(figsize=(12, 7))
    bars_p  = ax.bar(x - w, prec, w, label="Precision", color=_VIVID_PALETTE[0],
                     edgecolor="black", linewidth=0.8)
    bars_r  = ax.bar(x,     rec,  w, label="Recall",    color=_VIVID_PALETTE[1],
                     edgecolor="black", linewidth=0.8)
    bars_f1 = ax.bar(x + w, f1,   w, label="F1-Score",  color=_VIVID_PALETTE[2],
                     edgecolor="black", linewidth=0.8)

    # Annotate bars
    for bars in [bars_p, bars_r, bars_f1]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=_PLOT_STYLE["ANNOT_SIZE"] - 1, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(stage_names, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1-Score")
    ax.legend(frameon=True, shadow=True)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 8. False Alarm Rate Bar Chart ─────────────────────────────

def plot_false_alarm_rates(
    metrics: Dict,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/false_alarm_rates.png",
) -> None:
    """Bar chart of per-class false alarm rates."""
    far = [metrics["false_alarm_rate_per_class"].get(s, 0) for s in stage_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(stage_names, far, color=STAGE_COLORS,
                  edgecolor="black", linewidth=1.2, width=0.55)

    for bar, v in zip(bars, far):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=_PLOT_STYLE["ANNOT_SIZE"], fontweight="bold")

    ax.axhline(y=np.mean(far), color="#FF6F00", linestyle="--", linewidth=2.0,
               label=f"Mean FAR = {np.mean(far):.4f}")
    ax.set_ylabel("False Alarm Rate")
    ax.set_title("Per-Class False Alarm Rates")
    ax.legend(frameon=True, shadow=True)
    ax.set_ylim(0, max(far) * 1.35 + 0.005)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 9. Class Distribution Bar Chart ──────────────────────────

def plot_class_distribution(
    y: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/class_distribution.png",
    title: str = "Sleep Stage Distribution",
) -> None:
    """Bar chart showing count and percentage per class."""
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar([stage_names[int(c)] for c in classes],
                  counts, color=[STAGE_COLORS[int(c)] for c in classes],
                  edgecolor="black", linewidth=1.2, width=0.55)

    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, cnt + total * 0.008,
                f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=_PLOT_STYLE["ANNOT_SIZE"], fontweight="bold")

    ax.set_ylabel("Sample Count")
    ax.set_title(title)
    ax.set_ylim(0, max(counts) * 1.18)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 10. ROC Curves (per class, One-vs-Rest) ───────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/roc_curves.png",
) -> None:
    """One-vs-Rest ROC curves with AUC for each sleep stage."""
    n_classes = len(stage_names)
    fig, ax = plt.subplots(figsize=(10, 9))

    for i in range(n_classes):
        binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{stage_names[i]} (AUC={roc_auc:.3f})",
                linewidth=2.5, color=STAGE_COLORS[i])

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest")
    ax.legend(loc="lower right", frameon=True, shadow=True)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 11. Prediction Confidence Histogram ───────────────────────

def plot_confidence_distribution(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/confidence_distribution.png",
) -> None:
    """Histogram of max-class prediction confidence, split by correct/incorrect."""
    max_conf = np.max(y_prob, axis=1)
    correct  = (y_pred == y_true)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(max_conf[correct],  bins=50, alpha=0.75, color=_VIVID_PALETTE[2],
            edgecolor="black", linewidth=0.6, label="Correct", density=True)
    ax.hist(max_conf[~correct], bins=50, alpha=0.75, color=_VIVID_PALETTE[0],
            edgecolor="black", linewidth=0.6, label="Incorrect", density=True)

    ax.axvline(x=np.median(max_conf[correct]),  color=_VIVID_PALETTE[2],
               linestyle="--", linewidth=2.0,
               label=f"Correct median={np.median(max_conf[correct]):.3f}")
    ax.axvline(x=np.median(max_conf[~correct]) if (~correct).any() else 0,
               color=_VIVID_PALETTE[0], linestyle="--", linewidth=2.0,
               label=f"Incorrect median={np.median(max_conf[~correct]) if (~correct).any() else 0:.3f}")

    ax.set_xlabel("Max Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence — Correct vs Incorrect")
    ax.legend(frameon=True, shadow=True)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 12. Sleep Stage Transition Matrix Heatmap ─────────────────

def plot_transition_matrix(
    y_seq: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/transition_matrix.png",
    title: str = "Sleep Stage Transition Probabilities",
) -> None:
    """Heatmap of transition probabilities between consecutive stages."""
    n = len(stage_names)
    trans = np.zeros((n, n), dtype=float)
    for a, b in zip(y_seq[:-1], y_seq[1:]):
        trans[int(a), int(b)] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans / row_sums

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.YlGnBu
    im = ax.imshow(trans_prob, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition Probability", fontsize=_PLOT_STYLE["AXIS_LABEL_SIZE"] - 2)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(stage_names, fontweight="bold")
    ax.set_yticklabels(stage_names, fontweight="bold")
    ax.set_xlabel("To Stage")
    ax.set_ylabel("From Stage")
    ax.set_title(title)

    thresh = trans_prob.max() / 2.0
    for i in range(n):
        for j in range(n):
            colour = "white" if trans_prob[i, j] > thresh else "black"
            ax.text(j, i, f"{trans_prob[i, j]:.2f}",
                    ha="center", va="center", color=colour,
                    fontsize=_PLOT_STYLE["ANNOT_SIZE"] + 1, fontweight="bold")

    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 13. Feature Correlation Heatmap ───────────────────────────

def plot_feature_correlation(
    X_2d: np.ndarray,
    feature_names: List[str],
    save_path: str = "logs/feature_correlation.png",
) -> None:
    """Correlation heatmap of input features (pass 2-D flat array)."""
    corr = np.corrcoef(X_2d, rowvar=False)
    n = len(feature_names)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.8), max(10, n * 0.7)))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=_PLOT_STYLE["AXIS_LABEL_SIZE"] - 2)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short_names = [fn[:12] for fn in feature_names]   # truncate long names
    ax.set_xticklabels(short_names, rotation=55, ha="right",
                       fontsize=max(8, _PLOT_STYLE["TICK_SIZE"] - 3))
    ax.set_yticklabels(short_names,
                       fontsize=max(8, _PLOT_STYLE["TICK_SIZE"] - 3))
    ax.set_title("Feature Correlation Matrix")

    # Annotate only if matrix is small enough
    if n <= 22:
        for i in range(n):
            for j in range(n):
                colour = "white" if abs(corr[i, j]) > 0.55 else "black"
                ax.text(j, i, f"{corr[i, j]:.1f}",
                        ha="center", va="center", color=colour,
                        fontsize=max(6, _PLOT_STYLE["ANNOT_SIZE"] - 5))

    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 14. Stage Probability Over Time ──────────────────────────

def plot_stage_probabilities(
    y_prob: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/stage_probabilities.png",
    title: str = "",
    max_windows: int = 800,
) -> None:
    """Area / line plot of predicted class probabilities over time."""
    n = min(max_windows, len(y_prob))
    fig, ax = plt.subplots(figsize=(18, 6))
    t = np.arange(n)

    ax.stackplot(t, *[y_prob[:n, i] for i in range(len(stage_names))],
                 colors=STAGE_COLORS, alpha=0.7,
                 labels=stage_names)
    for i, name in enumerate(stage_names):
        ax.plot(t, y_prob[:n, i], color=STAGE_COLORS[i], linewidth=1.2, alpha=0.9)

    ax.set_xlim(0, n - 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Window Index (Time →)")
    ax.set_ylabel("Predicted Probability")
    ax.set_title(f"Stage Probability Over Time{' — ' + title if title else ''}")
    ax.legend(loc="upper right", ncol=4, frameon=True, shadow=True)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 15. Metrics Radar / Spider Chart ──────────────────────────

def plot_metrics_radar(
    metrics: Dict,
    save_path: str = "logs/metrics_radar.png",
) -> None:
    """Radar chart of key aggregate metrics."""
    labels = ["Accuracy", "Kappa", "Sleep Eff.", "1−FAR", "Sleep Quality"]
    values = [
        metrics.get("accuracy", 0),
        max(metrics.get("kappa", 0), 0),
        metrics.get("sleep_efficiency", 0),
        1 - metrics.get("mean_false_alarm_rate", 0),
        metrics.get("sleep_quality_score", 0) / 100,
    ]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color=_VIVID_PALETTE[1], alpha=0.25)
    ax.plot(angles, values, color=_VIVID_PALETTE[1], linewidth=2.5, marker="o",
            markersize=9, markerfacecolor=_VIVID_PALETTE[0])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=_PLOT_STYLE["AXIS_LABEL_SIZE"] - 2,
                       fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=_PLOT_STYLE["TICK_SIZE"] - 2)
    ax.set_title("Overall Performance Radar",
                 fontsize=_PLOT_STYLE["TITLE_SIZE"], fontweight="bold", pad=25)

    # Value annotations near each vertex
    for angle, val, label in zip(angles[:-1], values[:-1], labels):
        ax.annotate(f"{val:.3f}", xy=(angle, val),
                    fontsize=_PLOT_STYLE["ANNOT_SIZE"], fontweight="bold",
                    ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points")

    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 16. Per-Epoch Per-Class Accuracy Heatmap ──────────────────

def plot_epoch_class_accuracy(
    epoch_class_acc: np.ndarray,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/epoch_class_accuracy.png",
) -> None:
    """
    Heatmap of per-class accuracy at each epoch.
    epoch_class_acc: shape (n_epochs, n_classes)
    """
    fig, ax = plt.subplots(figsize=(max(10, len(stage_names) * 2.5),
                                    max(6, epoch_class_acc.shape[0] * 0.35)))
    cmap = plt.cm.YlGn
    im = ax.imshow(epoch_class_acc, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.03)
    cbar.set_label("Accuracy", fontsize=_PLOT_STYLE["AXIS_LABEL_SIZE"] - 2)

    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, fontweight="bold")
    ax.set_ylabel("Epoch")
    ax.set_title("Per-Class Accuracy Over Training Epochs")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 17. Sleep Quality Score Gauge ─────────────────────────────

def plot_sleep_quality_gauge(
    score: float,
    save_path: str = "logs/sleep_quality_gauge.png",
) -> None:
    """Half-circle gauge for the composite sleep quality score (0–100)."""
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(polar=True))

    # Draw background arcs
    theta_bg = np.linspace(np.pi, 0, 300)
    colors_bg = [(0.9, 0.2, 0.2), (1.0, 0.65, 0.0), (0.1, 0.8, 0.4)]
    bounds = [0, 50, 75, 100]
    for idx in range(3):
        low = bounds[idx] / 100 * np.pi
        high = bounds[idx + 1] / 100 * np.pi
        theta_seg = np.linspace(np.pi - low, np.pi - high, 100)
        ax.fill_between(theta_seg, 0.7, 1.0, color=colors_bg[idx], alpha=0.35)

    # Needle
    needle_angle = np.pi - (score / 100) * np.pi
    ax.plot([needle_angle, needle_angle], [0, 0.92], color="#1a1a2e",
            linewidth=3.5, solid_capstyle="round")
    ax.plot(needle_angle, 0.92, "o", color=_VIVID_PALETTE[0], markersize=10, zorder=5)

    ax.set_ylim(0, 1.1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines["polar"].set_visible(False)

    # Central text
    ax.text(np.pi / 2, 0.35, f"{score:.1f}",
            ha="center", va="center",
            fontsize=36, fontweight="bold", color="#1a1a2e",
            transform=ax.transData)
    ax.text(np.pi / 2, 0.08, "Sleep Quality",
            ha="center", va="center",
            fontsize=_PLOT_STYLE["AXIS_LABEL_SIZE"], fontweight="bold",
            color="#555555", transform=ax.transData)

    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 18. Metrics Summary Table Plot ────────────────────────────

def plot_metrics_table(
    metrics: Dict,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/metrics_summary_table.png",
) -> None:
    """Render a publication-quality summary table as an image."""
    # Build table data
    header = ["Stage", "Precision", "Recall", "F1-Score", "FAR"]
    rows = []
    for s in stage_names:
        pr = metrics["per_class"].get(s, {})
        far = metrics["false_alarm_rate_per_class"].get(s, 0)
        rows.append([
            s,
            f"{pr.get('precision', 0):.4f}",
            f"{pr.get('recall', 0):.4f}",
            f"{pr.get('f1-score', 0):.4f}",
            f"{far:.4f}",
        ])
    # Summary row
    rows.append([
        "Overall",
        f"{metrics.get('accuracy', 0):.4f}",
        f"κ={metrics.get('kappa', 0):.4f}",
        f"SQ={metrics.get('sleep_quality_score', 0):.1f}",
        f"{metrics.get('mean_false_alarm_rate', 0):.4f}",
    ])

    fig, ax = plt.subplots(figsize=(12, 2.5 + 0.5 * len(rows)))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=header,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(_PLOT_STYLE["ANNOT_SIZE"])
    table.scale(1, 1.8)

    # Style header
    for j in range(len(header)):
        cell = table[0, j]
        cell.set_facecolor("#1D7AF2")
        cell.set_text_props(color="white", fontweight="bold",
                            fontsize=_PLOT_STYLE["ANNOT_SIZE"] + 1)
    # Alternate row colours
    for i in range(1, len(rows) + 1):
        for j in range(len(header)):
            cell = table[i, j]
            cell.set_text_props(fontweight="bold")
            if i == len(rows):  # summary row
                cell.set_facecolor("#FFF3CD")
            elif i % 2 == 0:
                cell.set_facecolor("#F0F4FF")
            else:
                cell.set_facecolor("white")

    ax.set_title("Classification Metrics Summary",
                 fontsize=_PLOT_STYLE["TITLE_SIZE"], fontweight="bold", pad=20)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── 19. Multi-Plot Dashboard ─────────────────────────────────

def plot_dashboard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict,
    stage_names: List[str] = STAGE_NAMES,
    save_path: str = "logs/dashboard.png",
) -> None:
    """6-panel summary dashboard on a single figure."""
    fig = plt.figure(figsize=(28, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.30)

    # --- Panel 1: Confusion Matrix ---
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    im = ax1.imshow(cm, cmap=plt.cm.YlOrRd, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    n = len(stage_names)
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(stage_names, fontsize=11, fontweight="bold")
    ax1.set_yticklabels(stage_names, fontsize=11, fontweight="bold")
    for i in range(n):
        for j in range(n):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                     color=colour, fontsize=11, fontweight="bold")
    ax1.set_title("Confusion Matrix", fontsize=16, fontweight="bold")

    # --- Panel 2: Per-Class Metrics ---
    ax2 = fig.add_subplot(gs[0, 1])
    prec = [metrics["per_class"].get(s, {}).get("precision", 0) for s in stage_names]
    rec  = [metrics["per_class"].get(s, {}).get("recall", 0) for s in stage_names]
    f1   = [metrics["per_class"].get(s, {}).get("f1-score", 0) for s in stage_names]
    x = np.arange(n); w = 0.22
    ax2.bar(x - w, prec, w, label="Prec", color=_VIVID_PALETTE[0], edgecolor="black")
    ax2.bar(x, rec, w, label="Rec", color=_VIVID_PALETTE[1], edgecolor="black")
    ax2.bar(x + w, f1, w, label="F1", color=_VIVID_PALETTE[2], edgecolor="black")
    ax2.set_xticks(x); ax2.set_xticklabels(stage_names, fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.15); ax2.set_title("Per-Class Metrics", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=10, frameon=True)

    # --- Panel 3: ROC Curves ---
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(n):
        binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f"{stage_names[i]}({roc_auc:.2f})",
                 linewidth=2.2, color=STAGE_COLORS[i])
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.4); ax3.set_xlim(-0.02, 1.02)
    ax3.set_title("ROC Curves", fontsize=16, fontweight="bold")
    ax3.legend(fontsize=10, loc="lower right", frameon=True)

    # --- Panel 4: Confidence Distribution ---
    ax4 = fig.add_subplot(gs[1, 0])
    max_conf = np.max(y_prob, axis=1)
    correct = (y_pred == y_true)
    ax4.hist(max_conf[correct], bins=40, alpha=0.7, color=_VIVID_PALETTE[2],
             edgecolor="black", linewidth=0.5, label="Correct", density=True)
    ax4.hist(max_conf[~correct], bins=40, alpha=0.7, color=_VIVID_PALETTE[0],
             edgecolor="black", linewidth=0.5, label="Wrong", density=True)
    ax4.set_title("Confidence Distribution", fontsize=16, fontweight="bold")
    ax4.legend(fontsize=10, frameon=True)

    # --- Panel 5: FAR per class ---
    ax5 = fig.add_subplot(gs[1, 1])
    far = [metrics["false_alarm_rate_per_class"].get(s, 0) for s in stage_names]
    ax5.bar(stage_names, far, color=STAGE_COLORS, edgecolor="black", linewidth=1.0)
    ax5.axhline(np.mean(far), color="#FF6F00", linestyle="--", linewidth=2, label="Mean")
    ax5.set_title("False Alarm Rates", fontsize=16, fontweight="bold")
    ax5.legend(fontsize=10, frameon=True)

    # --- Panel 6: Radar ---
    ax6 = fig.add_subplot(gs[1, 2], polar=True)
    labels_r = ["Acc", "Kappa", "SleepEff", "1−FAR", "SQ"]
    vals_r = [
        metrics.get("accuracy", 0),
        max(metrics.get("kappa", 0), 0),
        metrics.get("sleep_efficiency", 0),
        1 - metrics.get("mean_false_alarm_rate", 0),
        metrics.get("sleep_quality_score", 0) / 100,
    ]
    angles_r = np.linspace(0, 2 * np.pi, len(labels_r), endpoint=False).tolist()
    vals_r += vals_r[:1]; angles_r += angles_r[:1]
    ax6.fill(angles_r, vals_r, color=_VIVID_PALETTE[1], alpha=0.25)
    ax6.plot(angles_r, vals_r, "o-", color=_VIVID_PALETTE[1], linewidth=2.2)
    ax6.set_xticks(angles_r[:-1])
    ax6.set_xticklabels(labels_r, fontsize=11, fontweight="bold")
    ax6.set_ylim(0, 1.05); ax6.set_title("Radar", fontsize=16, fontweight="bold", pad=20)

    # --- Panel 7-9 (bottom row): Hypnogram comparison ---
    ax_hyp_true = fig.add_subplot(gs[2, :])
    lim = min(500, len(y_true))
    t = np.arange(lim)
    ax_hyp_true.step(t, y_true[:lim], where="post", linewidth=1.8,
                     color="#1D7AF2", label="True", alpha=0.8)
    ax_hyp_true.step(t, y_pred[:lim], where="post", linewidth=1.8,
                     color="#E63946", label="Pred", alpha=0.8, linestyle="--")
    ax_hyp_true.set_yticks(range(n))
    ax_hyp_true.set_yticklabels(stage_names, fontsize=12, fontweight="bold")
    ax_hyp_true.invert_yaxis()
    ax_hyp_true.set_xlabel("Window Index", fontsize=14, fontweight="bold")
    ax_hyp_true.set_title("Hypnogram — True vs Predicted", fontsize=16, fontweight="bold")
    ax_hyp_true.legend(fontsize=12, frameon=True, shadow=True, loc="upper right")

    fig.suptitle("SleepNet — Evaluation Dashboard",
                 fontsize=26, fontweight="bold", y=1.01)
    _save_fig(fig, save_path)


# ── Master function: generate ALL plots ──────────────────────

def generate_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict,
    history=None,
    pareto_archive: Optional[List[dict]] = None,
    feature_names: Optional[List[str]] = None,
    X_flat: Optional[np.ndarray] = None,
    stage_names: List[str] = STAGE_NAMES,
    out_dir: str = "logs",
) -> None:
    """One-call convenience to produce every available plot."""
    d = str(Path(out_dir))
    Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("Generating comprehensive plot suite → %s/", d)

    # Core
    if history is not None:
        plot_training_history(history, f"{d}/training_curves.png")
    plot_confusion_matrix(y_true, y_pred, stage_names, f"{d}/confusion_matrix.png")
    plot_confusion_matrix_counts(y_true, y_pred, stage_names, f"{d}/confusion_matrix_counts.png")
    plot_hypnogram(y_pred[:min(600, len(y_pred))], stage_names,
                   f"{d}/hypnogram_predicted.png", "Predicted Sleep Hypnogram")
    plot_hypnogram_comparison(y_true, y_pred, stage_names, f"{d}/hypnogram_comparison.png")

    # Metrics
    plot_per_class_metrics(metrics, stage_names, f"{d}/per_class_metrics.png")
    plot_false_alarm_rates(metrics, stage_names, f"{d}/false_alarm_rates.png")
    plot_metrics_radar(metrics, f"{d}/metrics_radar.png")
    plot_metrics_table(metrics, stage_names, f"{d}/metrics_summary_table.png")

    # Probabilistic
    plot_roc_curves(y_true, y_prob, stage_names, f"{d}/roc_curves.png")
    plot_confidence_distribution(y_prob, y_pred, y_true, stage_names,
                                f"{d}/confidence_distribution.png")
    plot_stage_probabilities(y_prob, stage_names, f"{d}/stage_probabilities.png")

    # Distribution & transitions
    plot_class_distribution(y_true, stage_names, f"{d}/class_distribution_true.png",
                            "True Label Distribution")
    plot_class_distribution(y_pred, stage_names, f"{d}/class_distribution_pred.png",
                            "Predicted Label Distribution")
    plot_transition_matrix(y_pred, stage_names, f"{d}/transition_matrix_pred.png",
                           "Predicted Stage Transitions")
    plot_transition_matrix(y_true, stage_names, f"{d}/transition_matrix_true.png",
                           "True Stage Transitions")

    # Feature correlation (optional)
    if X_flat is not None and feature_names is not None:
        plot_feature_correlation(X_flat, feature_names, f"{d}/feature_correlation.png")

    # Sleep quality gauge
    if "sleep_quality_score" in metrics:
        plot_sleep_quality_gauge(metrics["sleep_quality_score"],
                                 f"{d}/sleep_quality_gauge.png")

    # Pareto (optional)
    if pareto_archive:
        plot_pareto_front(pareto_archive, f"{d}/pareto_front.png")

    # All-in-one dashboard
    plot_dashboard(y_true, y_pred, y_prob, metrics, stage_names, f"{d}/dashboard.png")

    logger.info("All %d plots saved to %s/", _count_pngs(d), d)


def _count_pngs(directory: str) -> int:
    return len(list(Path(directory).glob("*.png")))
