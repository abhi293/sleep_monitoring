#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from model import SparseFocalLoss
from preprocessing import FEATURE_COLS, STAGE_NAMES, apply_scaler, create_windows, load_raw, user_split
from src.utils import compute_metrics, generate_all_plots, save_metrics, setup_logging, sleep_quality_score

logger = logging.getLogger(__name__)

DATASET_PATH = HERE / "test_new.csv"
CHECKPOINT_DIR = HERE / "checkpoints"
WINDOW_SIZE = 30
STRIDE = 5
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
N_JOBS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pure MAResNet-BiGRU sleep-stage model")
    parser.add_argument("--model", default=str(CHECKPOINT_DIR / "best_model.keras"))
    parser.add_argument("--scaler", default=str(CHECKPOINT_DIR / "scaler.pkl"))
    parser.add_argument("--input", default=None, help="Optional CSV. Defaults to the built-in held-out test split.")
    parser.add_argument("--user_id", type=int, default=None)
    parser.add_argument("--out_dir", default=str(HERE / "evaluation_results"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--plots", choices=["full", "none"], default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.out_dir)
    logger.info("Evaluating pure MAResNet-BiGRU with standalone defaults.")

    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = tf.keras.models.load_model(model_path, custom_objects={"SparseFocalLoss": SparseFocalLoss})
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    if args.input:
        df = load_raw(args.input)
        tag = "custom_input"
    else:
        raw = load_raw(str(DATASET_PATH))
        _, _, df = user_split(raw, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
        tag = "test_split"

    if args.user_id is not None:
        df = df[df["User_ID"] == args.user_id].copy()
        tag = f"user_{args.user_id}"
    if df.empty:
        raise ValueError("No rows left after applying the selected input/user filter.")

    df = apply_scaler(df, scaler, FEATURE_COLS)
    X, y = create_windows(df, WINDOW_SIZE, STRIDE, FEATURE_COLS, n_jobs=N_JOBS)

    y_prob = model.predict(X, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = compute_metrics(y, y_pred, STAGE_NAMES)
    metrics["sleep_quality_score"] = sleep_quality_score(
        metrics["accuracy"],
        metrics["kappa"],
        metrics["mean_false_alarm_rate"],
        metrics["sleep_efficiency"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_csv(out_dir / f"{tag}_predictions.csv", index=False)
    pd.DataFrame(y_prob, columns=STAGE_NAMES).to_csv(out_dir / f"{tag}_probabilities.csv", index=False)
    save_metrics(metrics, str(out_dir / f"{tag}_metrics.json"))

    if args.plots == "full":
        generate_all_plots(
            y_true=y,
            y_pred=y_pred,
            y_prob=y_prob,
            metrics=metrics,
            feature_names=FEATURE_COLS,
            X_flat=X[:min(5000, len(X)), -1, :],
            stage_names=STAGE_NAMES,
            out_dir=str(out_dir),
        )

    logger.info("Evaluation complete: accuracy %.4f, kappa %.4f", metrics["accuracy"], metrics["kappa"])


if __name__ == "__main__":
    main()
