#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from model import SparseFocalLoss, SparseSmoothedCrossEntropy, build_model, print_model_summary
from preprocessing import (
    FEATURE_COLS,
    STAGE_NAMES,
    apply_scaler,
    create_windows,
    fit_scaler,
    get_class_weights,
    load_raw,
    user_split,
)
from src.utils import compute_metrics, generate_all_plots, save_metrics, setup_logging, sleep_quality_score

logger = logging.getLogger(__name__)

DATASET_PATH = HERE / "test_new.csv"
BASE_CHECKPOINT_DIR = HERE / "checkpoints"
BASE_LOG_DIR = HERE / "logs"

WINDOW_SIZE = 30
STRIDE = 10
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
N_JOBS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone CNN + optional BiLSTM + Transformer sleep model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--class_weights", action="store_true", help="Opt in to class weights; off by default.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a short data/model sanity check")

    bilstm = parser.add_mutually_exclusive_group()
    bilstm.add_argument("--with_bilstm", dest="use_bilstm", action="store_true", help="Run CNN+BiLSTM+Transformer")
    bilstm.add_argument("--without_bilstm", dest="use_bilstm", action="store_false", help="Run CNN+Transformer only")
    parser.set_defaults(use_bilstm=True)
    return parser.parse_args()


def variant_name(use_bilstm: bool) -> str:
    return "cnn_bilstm_transformer" if use_bilstm else "cnn_transformer"


def save_scaler(scaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def make_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(y), 5000), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_callbacks(checkpoint_dir: Path, log_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(log_dir / "training_log.csv"), append=False),
    ]


def prepare_data(window_size: int, stride: int):
    raw = load_raw(str(DATASET_PATH))
    train_df, val_df, test_df = user_split(raw, VAL_RATIO, TEST_RATIO, RANDOM_SEED)

    scaler = fit_scaler(train_df, FEATURE_COLS)
    train_df = apply_scaler(train_df, scaler, FEATURE_COLS)
    val_df = apply_scaler(val_df, scaler, FEATURE_COLS)
    test_df = apply_scaler(test_df, scaler, FEATURE_COLS)

    X_train, y_train = create_windows(train_df, window_size, stride, FEATURE_COLS, N_JOBS)
    X_val, y_val = create_windows(val_df, window_size, stride, FEATURE_COLS, N_JOBS)
    X_test, y_test = create_windows(test_df, window_size, stride, FEATURE_COLS, N_JOBS)
    return scaler, X_train, y_train, X_val, y_val, X_test, y_test


def main() -> None:
    args = parse_args()
    variant = variant_name(args.use_bilstm)
    checkpoint_dir = BASE_CHECKPOINT_DIR / variant
    log_dir = BASE_LOG_DIR / variant
    setup_logging(str(log_dir))

    logger.info("Training %s with TensorFlow %s", variant, tf.__version__)
    logger.info("Dataset: %s", DATASET_PATH)
    logger.info("Standalone defaults only. No config file and no optimizer-search path are used.")
    logger.info("Window size=%d stride=%d loss=%s class_weights=%s", args.window_size, args.stride, args.loss, args.class_weights)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t0 = time.time()
    scaler, X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(args.window_size, args.stride)
    save_scaler(scaler, checkpoint_dir / "scaler.pkl")

    if args.smoke_test:
        args.epochs = 2
        X_train, y_train = X_train[:1000], y_train[:1000]
        X_val, y_val = X_val[:300], y_val[:300]
        X_test, y_test = X_test[:300], y_test[:300]

    class_weights = get_class_weights(y_train, boost_minority=1.05) if args.class_weights else None
    logger.info(
        "Data ready in %.1fs: train=%s val=%s test=%s class_weights=%s",
        time.time() - t0,
        X_train.shape,
        X_val.shape,
        X_test.shape,
        class_weights,
    )

    model = build_model(
        window_size=X_train.shape[1],
        n_features=X_train.shape[2],
        use_bilstm=args.use_bilstm,
        learning_rate=args.learning_rate,
        class_weights=class_weights,
        loss_type=args.loss,
    )
    print_model_summary(model)

    history = model.fit(
        make_dataset(X_train, y_train, args.batch_size, shuffle=True),
        validation_data=make_dataset(X_val, y_val, args.batch_size, shuffle=False),
        epochs=args.epochs,
        callbacks=make_callbacks(checkpoint_dir, log_dir),
        verbose=1,
    )

    best_path = checkpoint_dir / "best_model.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(
            best_path,
            custom_objects={
                "SparseFocalLoss": SparseFocalLoss,
                "SparseSmoothedCrossEntropy": SparseSmoothedCrossEntropy,
            },
        )

    y_prob = model.predict(X_test, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(log_dir / "test_predictions.csv", index=False)
    pd.DataFrame(y_prob, columns=STAGE_NAMES).to_csv(log_dir / "test_probabilities.csv", index=False)

    metrics = compute_metrics(y_test, y_pred, STAGE_NAMES)
    metrics["variant"] = variant
    metrics["use_bilstm"] = bool(args.use_bilstm)
    metrics["sleep_quality_score"] = sleep_quality_score(
        metrics["accuracy"],
        metrics["kappa"],
        metrics["mean_false_alarm_rate"],
        metrics["sleep_efficiency"],
    )
    save_metrics(metrics, str(log_dir / "test_metrics.json"))

    generate_all_plots(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        metrics=metrics,
        history=history,
        feature_names=FEATURE_COLS,
        X_flat=X_test[:min(5000, len(X_test)), -1, :],
        stage_names=STAGE_NAMES,
        out_dir=str(log_dir),
    )

    logger.info(
        "Test accuracy %.4f | Kappa %.4f | Mean FAR %.4f | Sleep Quality %.2f",
        metrics["accuracy"],
        metrics["kappa"],
        metrics["mean_false_alarm_rate"],
        metrics["sleep_quality_score"],
    )


if __name__ == "__main__":
    main()
