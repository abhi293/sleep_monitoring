#!/usr/bin/env python3
"""
evaluate.py — Sleep Intelligence System: Inference & Evaluation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"

import tensorflow as tf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import (
    load_raw, user_split, apply_scaler, create_windows,
    FEATURE_COLS, STAGE_NAMES,
)

from src.utils import (
    setup_logging, compute_metrics, hrv_recovery_score,
    sleep_quality_score, load_scaler, save_metrics,
    plot_hypnogram, generate_all_plots,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sleep Model Evaluation & Inference")
    p.add_argument("--config", default="configs/model_config.yaml")
    p.add_argument("--model", default="checkpoints/best_model.keras")
    p.add_argument("--scaler", default="checkpoints/scaler.pkl")
    p.add_argument("--input", default=None)
    p.add_argument("--user_id", type=int, default=None)
    p.add_argument("--out_dir", default="logs")

    p.add_argument(
        "--plots",
        choices=["full", "minimal", "none"],
        default="full",
        help="Plot mode: full (default), minimal, none",
    )

    return p.parse_args()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def predict_sequence(model: tf.keras.Model, X: np.ndarray, batch_size: int = 256):
    y_prob = model.predict(X, batch_size=batch_size, verbose=0)
    return np.argmax(y_prob, axis=1), y_prob


def save_csv_outputs(out_dir: Path, tag: str, y_true, y_pred, y_prob):
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(out_dir / f"{tag}_predictions.csv", index=False)

    pd.DataFrame(y_prob).to_csv(
        out_dir / f"{tag}_probabilities.csv",
        index=False
    )


def save_sleep_quality_image(score: float, out_path: Path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("Sleep Quality Score")
    plt.text(0.5, 0.5, f"{score:.1f}/100",
             ha="center", va="center", fontsize=28)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def evaluate_and_report(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    tag: str,
    feature_cols,
    stage_names,
    plots_mode: str,
):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred, y_prob = predict_sequence(model, X)

    # Save CSV outputs
    save_csv_outputs(out_dir, tag, y, y_pred, y_prob)

    metrics = compute_metrics(y, y_pred, stage_names)

    if "RMSSD" in feature_cols:
        rmssd_idx = feature_cols.index("RMSSD")
        rmssd_seq = X[:, -1, rmssd_idx]
        metrics["hrv_recovery_score"] = hrv_recovery_score(rmssd_seq)
    else:
        metrics["hrv_recovery_score"] = None

    metrics["sleep_quality_score"] = sleep_quality_score(
        accuracy=metrics["accuracy"],
        kappa=metrics["kappa"],
        mean_far=metrics["mean_false_alarm_rate"],
        sleep_efficiency=metrics["sleep_efficiency"],
    )

    save_metrics(metrics, str(out_dir / f"{tag}_metrics.json"))

    # ───────── Plot handling ─────────

    if plots_mode == "full":
        logger.info("Generating FULL plot suite")
        generate_all_plots(
            y_true=y,
            y_pred=y_pred,
            y_prob=y_prob,
            metrics=metrics,
            history=None,
            pareto_archive=None,
            feature_names=feature_cols,
            X_flat=X[:min(5000, len(X)), -1, :],
            stage_names=stage_names,
            out_dir=out_dir,
        )

    elif plots_mode == "minimal":
        logger.info("Generating MINIMAL plots")
        
        

        save_sleep_quality_image(
            metrics["sleep_quality_score"],
            out_dir / f"{tag}_sleep_quality.png",
        )

    elif plots_mode == "none":
        logger.info("Plotting disabled")

    logger.info("Evaluation outputs saved → %s", out_dir)
    return metrics


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(log_dir=args.out_dir)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    feature_cols = cfg["data"]["feature_cols"]
    window_size  = cfg["data"]["window_size"]
    stride       = cfg["data"]["stride"]

    if not Path(args.model).exists():
        logger.error("Checkpoint not found: %s", args.model)
        sys.exit(1)

    if not Path(args.scaler).exists():
        logger.error("Scaler not found: %s", args.scaler)
        sys.exit(1)

    from src.model import SparseFocalLoss

    logger.info("Loading model...")
    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"SparseFocalLoss": SparseFocalLoss},
    )

    scaler = load_scaler(args.scaler)

    # ───────── Load data ─────────

    if args.input:
        df = load_raw(args.input)
        tag = "custom_input"
        if args.user_id is not None:
            df = df[df["User_ID"] == args.user_id].copy()
            tag = f"user_{args.user_id}"

    else:
        raw_df = load_raw(cfg["data"]["dataset_path"])
        _, _, df = user_split(
            raw_df,
            cfg["data"]["val_ratio"],
            cfg["data"]["test_ratio"],
            cfg["data"]["random_seed"],
        )
        tag = "test_split"
        if args.user_id is not None:
            df = df[df["User_ID"] == args.user_id].copy()
            tag = f"user_{args.user_id}_test"

    df_scaled = apply_scaler(df, scaler, feature_cols)
    X, y = create_windows(df_scaled, window_size, stride, feature_cols, n_jobs=4)

    if len(X) == 0:
        logger.error("No windows generated.")
        sys.exit(1)

    logger.info("Evaluating on %d windows", len(X))

    evaluate_and_report(
        model,
        X,
        y,
        args.out_dir,
        tag,
        feature_cols,
        STAGE_NAMES,
        args.plots,
    )


if __name__ == "__main__":
    main()
