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

DEFAULT_MODEL_CFG = {
    "num_classes": 4,
    "cnn_filters_1": 32,
    "cnn_filters_2": 64,
    "cnn_kernel_1": 3,
    "cnn_kernel_2": 3,
    "gru_units": 32,
    "lstm_units": 32,
    "dense_units": 64,
    "dropout": 0.40,
    "focal_gamma": 1.5,
}

DEFAULT_TRAINING_CFG = {
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 5e-4,
    "lr_schedule": "reduce_on_plateau",
    "early_stopping_patience": 8,
    "class_weights": True,
}

DEFAULT_DATA_CFG = {
    "dataset_path": "realistic_sleep_dataset_v4.csv",
    "feature_cols": FEATURE_COLS,
    "label_col": "Stage",
    "window_size": 30,
    "stride": 10,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
}

DEFAULT_MOPSO_CFG = {
    "n_particles": 8,
    "max_iter": 8,
    "n_jobs": 1,
    "quick_epochs": 3,
    "pareto_dir": "mopso_results",
}


# ────────────────────────────────────────────────────────────────
# Argument parser
# ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Hybrid CNN–GRU–LSTM Sleep Intelligence Model",
        epilog="""
MOPSO & Config Control:
  --mopso: Run MOPSO optimization (generates new config from optimization)
  --use_config_model: Load model/training hyperparams from YAML (only when --mopso not used)
  
Examples:
  # Ablation with built-in defaults (recommended for clean ablations)
  python train.py --without_gru
  
  # Ablation with custom data windowing
  python train.py --window_size 25 --stride 5 --with_lstm --without_gru
  
  # Run MOPSO optimization (generates new config)
  python train.py --mopso --mopso_iter 15 --mopso_particles 12
  
  # Use existing YAML config (e.g., after MOPSO has run once)
  python train.py --use_config_model --with_lstm
        """
    )
    p.add_argument("--config",          default="configs/model_config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--no_config",       action="store_true",
                   help="Do not load config YAML; use built-in standalone defaults.")
    p.add_argument("--dataset_path",    default=None,
                   help="Override dataset path")
    p.add_argument("--window_size",     type=int, default=None,
                   help="Override sliding-window size (default: 30 for ablations, 20 if --use_config_model)")
    p.add_argument("--stride",          type=int, default=None,
                   help="Override sliding-window stride (default: 10 for ablations, 3 if --use_config_model)")
    p.add_argument("--log_dir",         default=None,
                   help="Override base log directory")
    p.add_argument("--checkpoint_dir",  default=None,
                   help="Override base checkpoint directory")
    p.add_argument("--mopso",           action="store_true",
                   help="Run MOPSO hyperparameter optimization before training")
    p.add_argument("--mopso_iter",      type=int, default=None,
                   help="Override max MOPSO iterations")
    p.add_argument("--mopso_particles", type=int, default=None,
                   help="Override MOPSO swarm size")
    p.add_argument("--mopso_priority",  default="balanced",
                   choices=["accuracy", "efficiency", "balanced"],
                   help="How to select best MOPSO solution (default: balanced)")
    p.add_argument("--load_mopso",      action="store_true",
                   help="Load previous MOPSO results instead of re-running optimization")
    p.add_argument("--smoke_test",      action="store_true",
                   help="Quick 2-epoch test with reduced data")
    p.add_argument("--epochs",          type=int, default=None,
                   help="Override training epochs")
    p.add_argument("--batch_size",      type=int, default=None,
                   help="Override batch size")
    p.add_argument("--learning_rate",   type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--use_config_model", action="store_true",
                   help="Use model/training hyperparameters from YAML. "
                        "Ignored if --mopso is specified. Default for ablations: built-in defaults.")
    p.add_argument("--merge_val_for_final", action="store_true",
                   help="Merge train+val and monitor on an internal split. Off by default for honest ablations.")
    p.add_argument("--with_cnn",        dest="use_cnn", action="store_true",
                   help="Enable CNN encoder branch")
    p.add_argument("--without_cnn",     dest="use_cnn", action="store_false",
                   help="Disable CNN encoder branch")
    p.add_argument("--with_gru",        dest="use_gru", action="store_true",
                   help="Enable GRU branch")
    p.add_argument("--without_gru",     dest="use_gru", action="store_false",
                   help="Disable GRU branch")
    p.add_argument("--with_lstm",       dest="use_lstm", action="store_true",
                   help="Enable LSTM branch")
    p.add_argument("--without_lstm",    dest="use_lstm", action="store_false",
                   help="Disable LSTM branch")
    p.set_defaults(use_cnn=True, use_gru=True, use_lstm=True)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def default_config() -> dict:
    return {
        "data": DEFAULT_DATA_CFG.copy(),
        "model": DEFAULT_MODEL_CFG.copy(),
        "training": {
            **DEFAULT_TRAINING_CFG,
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
        },
        "mopso": DEFAULT_MOPSO_CFG.copy(),
    }


def ablation_variant(use_cnn: bool, use_gru: bool, use_lstm: bool) -> str:
    parts = []
    if use_cnn:
        parts.append("cnn")
    if use_gru:
        parts.append("gru")
    if use_lstm:
        parts.append("lstm")
    if not parts:
        raise ValueError("At least one component must be enabled.")
    return "_".join(parts)


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Logging & TF setup ─────────────────────────────────────
    cfg = default_config() if args.no_config else load_config(args.config)
    variant = ablation_variant(args.use_cnn, args.use_gru, args.use_lstm)
    
    # ── Determine config usage mode ────────────────────────────
    # KEY LOGIC:
    # 1. If --mopso: Ignore --use_config_model, will run MOPSO and use those results
    # 2. If NOT --mopso and --use_config_model: Use YAML config
    # 3. If NOT --mopso and NOT --use_config_model: Use built-in ablation defaults
    
    use_yaml_config = (not args.mopso) and args.use_config_model
    use_ablation_defaults = (not args.mopso) and (not args.use_config_model)
    
    # Set sensible defaults for data windowing based on mode
    # For ablations, use larger window/stride to minimize data leakage
    # For YAML config mode, use what's in the YAML (allow explicit windowing control)
    if use_ablation_defaults:
        # Ablation mode: use conservative windowing to avoid data leakage
        default_window_size = 30
        default_stride = 10
    elif use_yaml_config:
        # YAML mode: use what's configured in YAML
        default_window_size = cfg["data"].get("window_size", 20)
        default_stride = cfg["data"].get("stride", 3)
    else:
        # MOPSO mode: will be determined during MOPSO
        default_window_size = cfg["data"].get("window_size", 20)
        default_stride = cfg["data"].get("stride", 3)
    
    # Apply overrides from CLI (CLI always takes precedence)
    if args.window_size is not None:
        cfg["data"]["window_size"] = args.window_size
    elif use_ablation_defaults:
        # Only set ablation defaults if not already set
        cfg["data"]["window_size"] = default_window_size
        
    if args.stride is not None:
        cfg["data"]["stride"] = args.stride
    elif use_ablation_defaults:
        # Only set ablation defaults if not already set
        cfg["data"]["stride"] = default_stride

    if args.dataset_path is not None:
        cfg["data"]["dataset_path"] = args.dataset_path
    if args.log_dir is not None:
        cfg["training"]["log_dir"] = args.log_dir
    if args.checkpoint_dir is not None:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir

    base_log_dir = Path(cfg["training"]["log_dir"])
    base_ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    if args.mopso:
        log_dir = str(base_log_dir / f"mopso_{variant}")
        ckpt_dir = str(base_ckpt_dir / f"mopso_{variant}")
        cfg["mopso"]["pareto_dir"] = str(Path(cfg["mopso"]["pareto_dir"]) / variant)
    else:
        log_dir = str(base_log_dir / f"ablation_{variant}")
        ckpt_dir = str(base_ckpt_dir / f"ablation_{variant}")
    setup_logging(log_dir=log_dir)

    logging.getLogger("src.preprocessing").setLevel(logging.WARNING)

    configure_tf_cpu(n_cores=N_CORES)
    tf.config.threading.set_intra_op_parallelism_threads(N_CORES)
    tf.config.threading.set_inter_op_parallelism_threads(N_CORES)

    logger.info("=" * 80)
    logger.info("  Sleep Intelligence System -- Training")
    logger.info("  TF version: %s  |  Cores: %d", tf.__version__, N_CORES)
    logger.info("  Config mode: %s", 
                "MOPSO optimization" if args.mopso else 
                ("YAML config" if use_yaml_config else "Ablation defaults"))
    logger.info("  Data windowing: window_size=%d stride=%d", 
                cfg["data"]["window_size"], cfg["data"]["stride"])
    logger.info("  Ablation components: CNN=%s | GRU=%s | LSTM=%s | variant=%s",
                args.use_cnn, args.use_gru, args.use_lstm, variant)
    logger.info("=" * 80)

    # ── Override model/training config based on mode ──────────
    if use_ablation_defaults:
        cfg["model"] = DEFAULT_MODEL_CFG.copy()
        cfg["training"].update(DEFAULT_TRAINING_CFG)
        logger.info("[Ablation Mode] Using built-in model/training defaults.")
    elif use_yaml_config:
        logger.info("[YAML Config Mode] Using model/training hyperparameters from %s", args.config)
    elif args.mopso:
        logger.info("[MOPSO Mode] Will run optimization and use MOPSO-selected hyperparameters.")
    
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
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate

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
        "Data ready -> train: %s | val: %s | test: %s  (%.1fs)",
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
        df_tr_raw, df_va_raw, df_te_raw = user_split(
            raw_df,
            cfg["data"]["val_ratio"],
            cfg["data"]["test_ratio"],
            cfg["data"]["random_seed"],
        )
        scaler = data["scaler"]
        df_tr_scaled = apply_scaler(df_tr_raw, scaler, cfg["data"]["feature_cols"])
        df_va_scaled = apply_scaler(df_va_raw, scaler, cfg["data"]["feature_cols"])
        df_te_scaled = apply_scaler(df_te_raw, scaler, cfg["data"]["feature_cols"])

        mopso_data = {
            **data,
            "df_train": df_tr_scaled,
            "df_val":   df_va_scaled,
            "use_cnn": args.use_cnn,
            "use_gru": args.use_gru,
            "use_lstm": args.use_lstm,
        }

        n_iter      = args.mopso_iter      or cfg["mopso"]["max_iter"]
        n_particles = args.mopso_particles or cfg["mopso"]["n_particles"]
        mopso = MOPSO(
            n_particles=n_particles,
            max_iter=n_iter,
            n_jobs=cfg["mopso"]["n_jobs"],
            quick_epochs=cfg["mopso"]["quick_epochs"] if not args.smoke_test else 1,
            pareto_dir=cfg["mopso"]["pareto_dir"],
        )
        if args.load_mopso:
            logger.info("Loading existing MOPSO results (--load_mopso) …")
            mopso.load_results()
            pareto = mopso.pareto_archive
        else:
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
            logger.info("Re-windowing with MOPSO-selected window=%d, stride=%d",
                        window_size, cfg["data"]["stride"])
            from src.preprocessing import create_windows
            _stride = cfg["data"]["stride"]
            X_train, y_train = create_windows(df_tr_scaled, window_size, _stride,
                                               cfg["data"]["feature_cols"], N_CORES)
            X_val, y_val = create_windows(df_va_scaled, window_size, _stride,
                                           cfg["data"]["feature_cols"], N_CORES)
            X_test, y_test = create_windows(df_te_scaled, window_size, _stride,
                                            cfg["data"]["feature_cols"], N_CORES)

        plot_pareto_front(pareto, str(Path(cfg["mopso"]["pareto_dir"]) / "pareto_front.png"))
        logger.info("MOPSO best config: %s", model_cfg)
    else:
        logger.info("Stage 2/4 — MOPSO skipped (use --mopso to enable)")

    # ── Merge train + val for final model training ─────────────
    # Val set was needed during MOPSO (hyperparameter search) and for
    # early-stopping signal.  Now that hyperparameters are fixed, merging
    # train+val gives the model 85 % of the dataset instead of 70 %.
    # The held-out test set (15 %) is never touched until final evaluation.
    from src.preprocessing import get_class_weights
    if args.merge_val_for_final:
        logger.info(
            "Merging train+val for final training: %d + %d = %d windows",
            len(X_train), len(X_val), len(X_train) + len(X_val),
        )
        X_train = np.concatenate([X_train, X_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)
        perm = np.random.default_rng(cfg["data"]["random_seed"]).permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        if cfg["training"]["class_weights"]:
            class_weights = get_class_weights(y_train, boost_minority=1.5)
            logger.info("Recomputed class weights on merged set: %s", class_weights)
    else:
        logger.info("Keeping original validation split for ablation monitoring: train=%d val=%d",
                    len(X_train), len(X_val))

    # Model creation ─────────────────────────────────────────
    logger.info("Stage 3/4 — Building model …")
    model_cfg["use_cnn"] = args.use_cnn
    model_cfg["use_gru"] = args.use_gru
    model_cfg["use_lstm"] = args.use_lstm
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
        use_cnn=args.use_cnn,
        use_gru=args.use_gru,
        use_lstm=args.use_lstm,
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

    if args.merge_val_for_final:
        _monitor_ratio = 0.10
        _rng = np.random.default_rng(cfg["data"]["random_seed"] + 1)
        _monitor_idx = []
        for _cls in np.unique(y_train):
            _cls_idx = np.where(y_train == _cls)[0]
            _n = max(1, int(len(_cls_idx) * _monitor_ratio))
            _monitor_idx.extend(_rng.choice(_cls_idx, size=_n, replace=False).tolist())
        _monitor_idx = np.array(_monitor_idx)
        _train_mask = np.ones(len(X_train), dtype=bool)
        _train_mask[_monitor_idx] = False

        X_train_fit, y_train_fit = X_train[_train_mask], y_train[_train_mask]
        X_monitor, y_monitor = X_train[_monitor_idx], y_train[_monitor_idx]
        logger.info(
            "Monitor split: %d train / %d monitor (%.0f%% hold-out from merged set)",
            len(X_train_fit), len(X_monitor), _monitor_ratio * 100,
        )
        val_ds = make_tf_dataset(X_monitor, y_monitor, bs, shuffle=True, buffer=len(X_monitor))
    else:
        X_train_fit, y_train_fit = X_train, y_train
        logger.info("Using original validation set for monitoring: %d windows", len(X_val))
        val_ds = make_tf_dataset(X_val, y_val, bs, shuffle=False)

    train_ds = make_tf_dataset(X_train_fit, y_train_fit, bs, shuffle=True)

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
    metrics["ablation_variant"] = variant
    metrics["use_cnn"] = bool(args.use_cnn)
    metrics["use_gru"] = bool(args.use_gru)
    metrics["use_lstm"] = bool(args.use_lstm)

    # Recovery & quality score
    # Use SVM (movement) proxy as a stand-in for RMSSD if not segmented
    sqs = sleep_quality_score(
        accuracy=metrics["accuracy"],
        kappa=metrics["kappa"],
        mean_far=metrics["mean_false_alarm_rate"],
        sleep_efficiency=metrics["sleep_efficiency"],
    )
    metrics["sleep_quality_score"] = sqs

    logger.info("=" * 60)
    logger.info("  TEST RESULTS")
    logger.info("  Accuracy        : %.4f", metrics["accuracy"])
    logger.info("  Cohen's Kappa   : %.4f", metrics["kappa"])
    logger.info("  Mean FAR        : %.4f", metrics["mean_false_alarm_rate"])
    logger.info("  Sleep Efficiency: %.4f", metrics["sleep_efficiency"])
    logger.info("  Sleep Quality ^ : %.2f / 100", metrics["sleep_quality_score"])
    logger.info("=" * 60)

    # Per-stage metrics
    for stage in data["stage_names"]:
        pr = metrics["per_class"].get(stage, {})
        logger.info("  %-6s -> P=%.3f R=%.3f F1=%.3f  FAR=%.3f",
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
    logger.info("Done.")


if __name__ == "__main__":
    main()
