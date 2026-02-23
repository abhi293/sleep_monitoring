#!/usr/bin/env python3
"""
train.py — Sleep Intelligence System: Hybrid CNN–GRU–LSTM Training
═══════════════════════════════════════════════════════════════════

Workflow
────────
1. Load & preprocess realistic_sleep_dataset_v4.csv
2. User-stratified train / val / test split (no data leakage)
3. [Optional] MOPSO hyperparameter optimisation (--mopso flag)
4. Build Hybrid CNN–GRU–LSTM model
5. Train with full 4-core CPU parallelism and class-weight balancing
6. Save best checkpoint, scaler, metrics and diagnostic plots

Usage
─────
  # Standard training with default config
  python train.py

  # Custom config
  python train.py --config configs/model_config.yaml

  # With MOPSO optimisation before training
  python train.py --mopso --mopso_iter 20 --mopso_particles 15

  # Quick smoke-test (2 epochs, small data)
  python train.py --smoke_test
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Force multi-core CPU usage before importing TF ──────────────
N_CORES = 4
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["OMP_NUM_THREADS"]       = str(N_CORES)
os.environ["MKL_NUM_THREADS"]       = str(N_CORES)

import numpy as np
import yaml
import tensorflow as tf

# Project modules
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing import full_pipeline, FEATURE_COLS
from src.model import build_model, build_from_config, print_model_summary
from src.mopso import MOPSO
from src.utils import (
    setup_logging, configure_tf_cpu,
    get_callbacks, compute_metrics, hrv_recovery_score,
    sleep_quality_score, save_scaler, save_metrics,
    plot_training_history, plot_confusion_matrix,
    plot_hypnogram, plot_pareto_front,
    generate_all_plots, plot_class_distribution,
    STAGE_NAMES,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Argument parser
# ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Hybrid CNN–GRU–LSTM Sleep Intelligence Model"
    )
    p.add_argument("--config",          default="configs/model_config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--mopso",           action="store_true",
                   help="Run MOPSO optimisation before training")
    p.add_argument("--mopso_iter",      type=int, default=None,
                   help="Override max MOPSO iterations")
    p.add_argument("--mopso_particles", type=int, default=None,
                   help="Override MOPSO swarm size")
    p.add_argument("--mopso_priority",  default="balanced",
                   choices=["accuracy", "efficiency", "balanced"],
                   help="How to select best MOPSO solution (default: balanced)")
    p.add_argument("--smoke_test",      action="store_true",
                   help="Quick 2-epoch test with reduced data")
    p.add_argument("--epochs",          type=int, default=None,
                   help="Override training epochs")
    p.add_argument("--batch_size",      type=int, default=None,
                   help="Override batch size")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Logging & TF setup ─────────────────────────────────────
    cfg = load_config(args.config)
    log_dir   = cfg["training"]["log_dir"]
    ckpt_dir  = cfg["training"]["checkpoint_dir"]
    setup_logging(log_dir=log_dir)
    configure_tf_cpu(n_cores=N_CORES)
    tf.config.threading.set_intra_op_parallelism_threads(N_CORES)
    tf.config.threading.set_inter_op_parallelism_threads(N_CORES)

    logger.info("═" * 60)
    logger.info("  Sleep Intelligence System — Training")
    logger.info("  TF version: %s  |  Cores: %d", tf.__version__, N_CORES)
    logger.info("═" * 60)

    # ── Override config with CLI args ──────────────────────────
    if args.smoke_test:
        cfg["training"]["epochs"] = 2
        cfg["training"]["batch_size"] = 256
        cfg["data"]["window_size"] = 20
        cfg["data"]["stride"] = 10
        cfg["mopso"]["n_particles"] = 4
        cfg["mopso"]["max_iter"] = 2
        logger.info("Smoke-test mode: reduced epochs/data")

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    # ── Preprocessing ──────────────────────────────────────────
    t0 = time.time()
    logger.info("Stage 1/4 — Preprocessing …")

    data = full_pipeline(
        dataset_path=cfg["data"]["dataset_path"],
        window_size=cfg["data"]["window_size"],
        stride=cfg["data"]["stride"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        feature_cols=cfg["data"]["feature_cols"],
        seed=cfg["data"]["random_seed"],
        n_jobs=N_CORES,
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    class_weights    = data["class_weights"] if cfg["training"]["class_weights"] else None

    save_scaler(data["scaler"], str(Path(ckpt_dir) / "scaler.pkl"))

    n_features  = X_train.shape[2]
    window_size = X_train.shape[1]

    logger.info(
        "Data ready → train: %s | val: %s | test: %s  (%.1fs)",
        X_train.shape, X_val.shape, X_test.shape, time.time() - t0
    )

    # ── Smoke-test: subsample ──────────────────────────────────
    if args.smoke_test:
        max_n = 2000
        X_train = X_train[:max_n]; y_train = y_train[:max_n]
        X_val   = X_val[:500];     y_val   = y_val[:500]
        X_test  = X_test[:500];    y_test  = y_test[:500]
        logger.info("Smoke-test: using %d/%d/%d samples", len(X_train), len(X_val), len(X_test))

    # ── MOPSO (optional) ───────────────────────────────────────
    model_cfg = cfg["model"].copy()

    if args.mopso:
        logger.info("Stage 2/4 — MOPSO hyperparameter optimisation …")

        # We need raw DFs for re-windowing inside MOPSO workers
        from src.preprocessing import load_raw, user_split, apply_scaler, fit_scaler
        raw_df = load_raw(cfg["data"]["dataset_path"])
        df_tr_raw, df_va_raw, _ = user_split(
            raw_df,
            cfg["data"]["val_ratio"],
            cfg["data"]["test_ratio"],
            cfg["data"]["random_seed"],
        )
        scaler = data["scaler"]
        df_tr_scaled = apply_scaler(df_tr_raw, scaler, cfg["data"]["feature_cols"])
        df_va_scaled = apply_scaler(df_va_raw, scaler, cfg["data"]["feature_cols"])

        mopso_data = {
            **data,
            "df_train": df_tr_scaled,
            "df_val":   df_va_scaled,
        }

        n_iter      = args.mopso_iter      or cfg["mopso"]["max_iter"]
        n_particles = args.mopso_particles or cfg["mopso"]["n_particles"]
        mopso = MOPSO(
            n_particles=n_particles,
            max_iter=n_iter,
            n_jobs=N_CORES,
            quick_epochs=5 if not args.smoke_test else 2,
            pareto_dir=cfg["mopso"]["pareto_dir"],
        )
        pareto = mopso.optimize(mopso_data, n_features=n_features)
        best_hp = mopso.best_hyperparams(priority=args.mopso_priority)

        # Update model_cfg with MOPSO-found values
        for k in ["cnn_filters_1", "cnn_filters_2", "gru_units", "lstm_units",
                  "dense_units", "dropout"]:
            if k in best_hp:
                model_cfg[k] = best_hp[k]
        cfg["training"]["learning_rate"] = best_hp.get("learning_rate",
                                                        cfg["training"]["learning_rate"])
        window_size = int(best_hp.get("window_size", window_size))

        # Re-window if MOPSO changed window_size
        if window_size != X_train.shape[1]:
            logger.info("Re-windowing with MOPSO-selected window=%d", window_size)
            from src.preprocessing import create_windows
            X_train, y_train = create_windows(df_tr_scaled, window_size,
                                               window_size // 3,
                                               cfg["data"]["feature_cols"], N_CORES)
            X_val, y_val = create_windows(df_va_scaled, window_size,
                                           window_size // 3,
                                           cfg["data"]["feature_cols"], N_CORES)

        plot_pareto_front(pareto, str(Path(cfg["mopso"]["pareto_dir"]) / "pareto_front.png"))
        logger.info("MOPSO best config: %s", model_cfg)
    else:
        logger.info("Stage 2/4 — MOPSO skipped (use --mopso to enable)")

    # ── Model creation ─────────────────────────────────────────
    logger.info("Stage 3/4 — Building model …")
    model = build_model(
        window_size=window_size,
        n_features=n_features,
        num_classes=model_cfg.get("num_classes", 4),
        cnn_filters_1=model_cfg.get("cnn_filters_1", 64),
        cnn_filters_2=model_cfg.get("cnn_filters_2", 128),
        cnn_kernel_1=model_cfg.get("cnn_kernel_1", 3),
        cnn_kernel_2=model_cfg.get("cnn_kernel_2", 3),
        gru_units=model_cfg.get("gru_units", 64),
        lstm_units=model_cfg.get("lstm_units", 64),
        dense_units=model_cfg.get("dense_units", 128),
        dropout=float(model_cfg.get("dropout", 0.30)),
        learning_rate=float(cfg["training"].get("learning_rate", 1e-3)),
        focal_gamma=float(model_cfg.get("focal_gamma", 2.0)),
        class_weights=class_weights,
    )
    print_model_summary(model)

    # ── Training ───────────────────────────────────────────────
    logger.info("Stage 4/4 — Training …")
    callbacks = get_callbacks(
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
        patience=cfg["training"]["early_stopping_patience"],
        lr_schedule=cfg["training"]["lr_schedule"],
        epochs=cfg["training"]["epochs"],
        learning_rate=float(cfg["training"]["learning_rate"]),
    )

    # TF Dataset pipeline for efficient multi-threaded CPU feeding
    def make_tf_dataset(X: np.ndarray, y: np.ndarray,
                        batch_size: int, shuffle: bool = False,
                        buffer: int = 5000) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=buffer, seed=42)
        ds = (ds.batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
        return ds

    bs = cfg["training"]["batch_size"]
    train_ds = make_tf_dataset(X_train, y_train, bs, shuffle=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   bs, shuffle=False)

    t_train = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks,
        # class_weight is already baked into focal loss; no need to pass again
        verbose=1,
    )
    train_time = time.time() - t_train
    logger.info("Training completed in %.1fs", train_time)

    # ── Load best checkpoint ───────────────────────────────────
    best_path = str(Path(ckpt_dir) / "best_model.keras")
    if Path(best_path).exists():
        from src.model import SparseFocalLoss
        model = tf.keras.models.load_model(
            best_path,
            custom_objects={"SparseFocalLoss": SparseFocalLoss},
        )
        logger.info("Loaded best checkpoint: %s", best_path)

    # ── Evaluation ─────────────────────────────────────────────
    logger.info("Evaluating on test set …")
    y_pred_prob = model.predict(X_test, batch_size=bs, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    metrics = compute_metrics(y_test, y_pred, data["stage_names"])
    metrics["training_time_sec"] = round(train_time, 2)

    # Recovery & quality score
    # Use SVM (movement) proxy as a stand-in for RMSSD if not segmented
    sqs = sleep_quality_score(
        accuracy=metrics["accuracy"],
        kappa=metrics["kappa"],
        mean_far=metrics["mean_false_alarm_rate"],
        sleep_efficiency=metrics["sleep_efficiency"],
    )
    metrics["sleep_quality_score"] = sqs

    logger.info("═" * 60)
    logger.info("  TEST RESULTS")
    logger.info("  Accuracy        : %.4f", metrics["accuracy"])
    logger.info("  Cohen's Kappa   : %.4f", metrics["kappa"])
    logger.info("  Mean FAR        : %.4f", metrics["mean_false_alarm_rate"])
    logger.info("  Sleep Efficiency: %.4f", metrics["sleep_efficiency"])
    logger.info("  Sleep Quality ↑ : %.2f / 100", metrics["sleep_quality_score"])
    logger.info("═" * 60)

    # Per-stage metrics
    for stage in data["stage_names"]:
        pr = metrics["per_class"].get(stage, {})
        logger.info("  %-6s → P=%.3f R=%.3f F1=%.3f  FAR=%.3f",
                    stage,
                    pr.get("precision", 0),
                    pr.get("recall", 0),
                    pr.get("f1-score", 0),
                    metrics["false_alarm_rate_per_class"].get(stage, 0))

    save_metrics(metrics, str(Path(log_dir) / "test_metrics.json"))

    # ── Plots ──────────────────────────────────────────────────
    # Pre-training class distribution (using raw train labels)
    plot_class_distribution(
        y_train, data["stage_names"],
        str(Path(log_dir) / "class_distribution_train.png"),
        "Training Set — Class Distribution",
    )

    # Build flat feature matrix for correlation plot (subsample to save memory)
    subsample_n = min(5000, len(X_test))
    X_flat = X_test[:subsample_n].reshape(subsample_n, -1)
    # Use tiled feature names to match the flattened window
    feat_names_flat = [f"{fn}_t{t}" for t in range(X_test.shape[1])
                       for fn in cfg["data"]["feature_cols"]]

    # Generate full comprehensive plot suite
    generate_all_plots(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_pred_prob,
        metrics=metrics,
        history=history,
        pareto_archive=None,
        feature_names=cfg["data"]["feature_cols"],
        X_flat=X_test[:subsample_n, -1, :],   # last timestep per window
        stage_names=data["stage_names"],
        out_dir=log_dir,
    )

    logger.info("All outputs saved to '%s/' and '%s/'", log_dir, ckpt_dir)
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
