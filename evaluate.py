#!/usr/bin/env python3
"""
evaluate.py — Sleep Intelligence System: Inference & Evaluation
═══════════════════════════════════════════════════════════════

Loads a trained checkpoint and runs inference on:
  • the test split (held-out users)
  • or a user-supplied raw CSV file

Produces:
  • Per-stage precision / recall / F1 / FAR table
  • Cohen's kappa, sleep efficiency, sleep quality score
  • HRV recovery score
  • Confusion matrix, hypnogram, and feature importance plot
  • JSON report

Usage
─────
  # Evaluate on test split from the original dataset
  python evaluate.py

  # Evaluate a specific user's raw CSV
  python evaluate.py --input path/to/user_session.csv --user_id 5

  # Specify a different checkpoint
  python evaluate.py --model checkpoints/best_model.keras

  # Save report to custom location
  python evaluate.py --out_dir results/user_5
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

sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing import (
    load_raw, user_split, apply_scaler, create_windows,
    FEATURE_COLS, STAGE_NAMES,
)
from src.utils import (
    setup_logging, compute_metrics, hrv_recovery_score,
    sleep_quality_score, load_scaler, save_metrics,
    plot_confusion_matrix, plot_hypnogram, plot_pareto_front,
    generate_all_plots, plot_stage_probabilities,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Argument parser
# ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sleep Model Evaluation & Inference")
    p.add_argument("--config",   default="configs/model_config.yaml")
    p.add_argument("--model",    default="checkpoints/best_model.keras",
                   help="Path to trained .keras model checkpoint")
    p.add_argument("--scaler",   default="checkpoints/scaler.pkl",
                   help="Path to saved RobustScaler")
    p.add_argument("--input",    default=None,
                   help="Optional: path to a raw CSV file for inference")
    p.add_argument("--user_id",  type=int, default=None,
                   help="If --input is the main dataset, evaluate only this user")
    p.add_argument("--out_dir",  default="logs",
                   help="Directory to save evaluation outputs")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def predict_sequence(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred_class, y_pred_prob)."""
    y_prob = model.predict(X, batch_size=batch_size, verbose=0)
    return np.argmax(y_prob, axis=1), y_prob


def evaluate_and_report(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    tag: str = "test",
    feature_cols: list = FEATURE_COLS,
    stage_names: list = STAGE_NAMES,
) -> dict:
    """Full evaluation → metrics, plots, JSON report."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    y_pred, y_prob = predict_sequence(model, X)

    metrics = compute_metrics(y, y_pred, stage_names)

    # HRV recovery: if RMSSD is a feature, compute proxy over the session
    if "RMSSD" in feature_cols:
        rmssd_idx = feature_cols.index("RMSSD")
        rmssd_seq = X[:, -1, rmssd_idx]   # last timestep per window
        metrics["hrv_recovery_score"] = hrv_recovery_score(rmssd_seq)
    else:
        metrics["hrv_recovery_score"] = None

    sqs = sleep_quality_score(
        accuracy=metrics["accuracy"],
        kappa=metrics["kappa"],
        mean_far=metrics["mean_false_alarm_rate"],
        sleep_efficiency=metrics["sleep_efficiency"],
    )
    metrics["sleep_quality_score"] = sqs

    # ── Print report ────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  Sleep Evaluation Report -- {tag.upper()}")
    print("=" * 64)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Cohen's Kappa     : {metrics['kappa']:.4f}")
    print(f"  Mean FAR          : {metrics['mean_false_alarm_rate']:.4f}")
    print(f"  Sleep Efficiency  : {metrics['sleep_efficiency']:.4f}")
    if metrics["hrv_recovery_score"] is not None:
        print(f"  HRV Recovery Scr  : {metrics['hrv_recovery_score']:.4f}")
    print(f"  Sleep Quality ^   : {metrics['sleep_quality_score']:.2f} / 100")
    print("-" * 64)
    print(f"  {'Stage':<8}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}  {'FAR':>7}")
    print("-" * 64)
    for stage in stage_names:
        pr = metrics["per_class"].get(stage, {})
        far = metrics["false_alarm_rate_per_class"].get(stage, 0)
        print(f"  {stage:<8}  {pr.get('precision',0):>9.4f}  "
              f"{pr.get('recall',0):>7.4f}  {pr.get('f1-score',0):>6.4f}  {far:>7.4f}")
    print("=" * 64 + "\n")

    # ── Save outputs ────────────────────────────────────────────
    save_metrics(metrics, str(Path(out_dir) / f"{tag}_metrics.json"))

    # Comprehensive plot suite
    generate_all_plots(
        y_true=y,
        y_pred=y_pred,
        y_prob=y_prob,
        metrics=metrics,
        history=None,
        pareto_archive=None,
        feature_names=feature_cols,
        X_flat=X[:min(5000, len(X)), -1, :],   # last timestep per window
        stage_names=stage_names,
        out_dir=out_dir,
    )

    logger.info("Evaluation report saved to %s/", out_dir)
    return metrics



# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(log_dir=args.out_dir)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    feature_cols = cfg["data"]["feature_cols"]
    window_size  = cfg["data"]["window_size"]
    stride       = cfg["data"]["stride"]

    # ── Load model & scaler ────────────────────────────────────
    if not Path(args.model).exists():
        logger.error("Checkpoint not found: %s  (run train.py first)", args.model)
        sys.exit(1)
    if not Path(args.scaler).exists():
        logger.error("Scaler not found: %s  (run train.py first)", args.scaler)
        sys.exit(1)

    logger.info("Loading model: %s", args.model)
    from src.model import SparseFocalLoss
    model  = tf.keras.models.load_model(
        args.model,
        custom_objects={"SparseFocalLoss": SparseFocalLoss},
    )
    scaler = load_scaler(args.scaler)

    # ── Determine data to evaluate ─────────────────────────────
    if args.input is not None:
        # Inference on a new CSV file
        df = load_raw(args.input)
        if args.user_id is not None:
            df = df[df["User_ID"] == args.user_id].copy()
            tag = f"user_{args.user_id}"
        else:
            tag = "custom_input"
        df_scaled = apply_scaler(df, scaler, feature_cols)
        X, y = create_windows(df_scaled, window_size, stride, feature_cols, n_jobs=4)
    else:
        # Use test split of the original dataset
        logger.info("Loading original dataset for test-split evaluation …")
        raw_df = load_raw(cfg["data"]["dataset_path"])
        _, _, df_test = user_split(
            raw_df,
            cfg["data"]["val_ratio"],
            cfg["data"]["test_ratio"],
            cfg["data"]["random_seed"],
        )
        if args.user_id is not None:
            df_test = df_test[df_test["User_ID"] == args.user_id].copy()
            tag = f"user_{args.user_id}_test"
        else:
            tag = "test_split"
        df_scaled = apply_scaler(df_test, scaler, feature_cols)
        X, y = create_windows(df_scaled, window_size, stride, feature_cols, n_jobs=4)

    if len(X) == 0:
        logger.error("No windows generated. Check input data / window_size.")
        sys.exit(1)

    logger.info("Evaluating on %d windows (shape %s) …", len(X), X.shape)
    evaluate_and_report(model, X, y, args.out_dir, tag, feature_cols)


if __name__ == "__main__":
    main()
