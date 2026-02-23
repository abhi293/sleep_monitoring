"""
model.py
────────────────────────────────────────────────────────────────
Hybrid CNN – GRU – LSTM architecture for sleep stage classification.

Architecture overview
─────────────────────
Input: [batch, T, F]  (T = window_size, F = n_features)

   ┌─────────────────────────────────────────────┐
   │  CNN Branch (local pattern extraction)       │
   │  Conv1D(f1,k) → BN → ReLU                   │
   │  Conv1D(f2,k) → BN → ReLU → MaxPool         │
   │  Conv1D(f2,k) → BN → ReLU → GlobalAvgPool   │
   └──────────────────────┬──────────────────────┘
                          │ CNN feature map [B, T', f2]
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    ┌────▼────┐     ┌────▼────┐           │
    │Bidirec. │     │Bidirec. │           │
    │  GRU    │     │  LSTM   │           │
    │(short)  │     │(long)   │           │
    └────┬────┘     └────┬────┘           │
         │               │        GlobalAvgPool
         └────────┬───────┘          [B, f2]
                  │
           Concat → Dense(d) → Dropout → Dense(4, softmax)

Notes
─────
• All three branches operate on the same CNN feature map so the CNN
  acts as a shared encoder that removes raw-signal noise before
  sequential processing.
• GRU captures short-disturbance dynamics; LSTM captures sleep-cycle
  level dynamics.
• Bidirectional wrappers improve sensitivity for transitions.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, MaxPooling1D,
    GlobalAveragePooling1D, Bidirectional, GRU, LSTM,
    Dense, Dropout, Concatenate, Reshape, LayerNormalization,
    MultiHeadAttention, Add,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ────────────────────────────────────────────────────────────────
# Building blocks
# ────────────────────────────────────────────────────────────────

def _conv_block(x: tf.Tensor, filters: int, kernel: int,
                pool: bool = False, l2_reg: float = 1e-4) -> tf.Tensor:
    """Conv1D → BN → ReLU → optional MaxPool."""
    x = Conv1D(filters, kernel_size=kernel, padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if pool:
        x = MaxPooling1D(pool_size=2, padding="same")(x)
    return x


def _residual_conv_block(x: tf.Tensor, filters: int, kernel: int,
                          l2_reg: float = 1e-4) -> tf.Tensor:
    """Residual Conv1D block for deeper CNN paths."""
    shortcut = x
    x = _conv_block(x, filters, kernel)
    x = _conv_block(x, filters, kernel)
    # Project shortcut if shapes differ
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same",
                          kernel_regularizer=l2(l2_reg))(shortcut)
    return Add()([x, shortcut])


# ────────────────────────────────────────────────────────────────
# Main model builder
# ────────────────────────────────────────────────────────────────

def build_model(
    window_size: int = 30,
    n_features: int = 11,
    num_classes: int = 4,
    cnn_filters_1: int = 64,
    cnn_filters_2: int = 128,
    cnn_kernel_1: int = 3,
    cnn_kernel_2: int = 3,
    cnn_pool_size: int = 2,
    gru_units: int = 64,
    lstm_units: int = 64,
    dense_units: int = 128,
    dropout: float = 0.30,
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-4,
) -> Model:
    """
    Build and compile the Hybrid CNN–GRU–LSTM model.

    Parameters
    ----------
    window_size : sequence length (time steps)
    n_features  : number of input features per step
    num_classes : number of sleep-stage output classes (Awake/Light/Deep/REM)
    ...         : architecture hyper-parameters (tunable by MOPSO)

    Returns
    -------
    Compiled tf.keras.Model
    """
    inp = Input(shape=(window_size, n_features), name="input_sequence")

    # ── CNN encoder ────────────────────────────────────────────
    cnn = _conv_block(inp, cnn_filters_1, cnn_kernel_1, pool=False, l2_reg=l2_reg)
    cnn = _residual_conv_block(cnn, cnn_filters_1, cnn_kernel_1, l2_reg=l2_reg)
    cnn = _conv_block(cnn, cnn_filters_2, cnn_kernel_2, pool=True,  l2_reg=l2_reg)
    cnn = _residual_conv_block(cnn, cnn_filters_2, cnn_kernel_2, l2_reg=l2_reg)
    # cnn → [batch, T//2, cnn_filters_2]

    # ── GRU branch (short-term disturbances) ───────────────────
    gru_out = Bidirectional(
        GRU(gru_units, return_sequences=True, dropout=0.1,
            recurrent_dropout=0.0,
            kernel_regularizer=l2(l2_reg)),
        name="bidirectional_gru"
    )(cnn)                                         # [B, T', 2*gru_units]
    gru_out = LayerNormalization()(gru_out)
    gru_pooled = GlobalAveragePooling1D(name="gru_pool")(gru_out)   # [B, 2*gru_units]

    # ── LSTM branch (long-term sleep cycles) ───────────────────
    lstm_out = Bidirectional(
        LSTM(lstm_units, return_sequences=True, dropout=0.1,
             recurrent_dropout=0.0,
             kernel_regularizer=l2(l2_reg)),
        name="bidirectional_lstm"
    )(cnn)                                         # [B, T', 2*lstm_units]
    lstm_out = LayerNormalization()(lstm_out)
    lstm_pooled = GlobalAveragePooling1D(name="lstm_pool")(lstm_out) # [B, 2*lstm_units]

    # ── CNN global context ─────────────────────────────────────
    cnn_pooled = GlobalAveragePooling1D(name="cnn_pool")(cnn)       # [B, cnn_filters_2]

    # ── Merge all branches ─────────────────────────────────────
    merged = Concatenate(name="merge")([cnn_pooled, gru_pooled, lstm_pooled])
    # merged → [B, cnn_filters_2 + 2*gru_units + 2*lstm_units]

    # ── Classification head ────────────────────────────────────
    x = Dense(dense_units, activation="relu",
              kernel_regularizer=l2(l2_reg), name="head_dense_1")(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout, name="head_dropout")(x)
    x = Dense(dense_units // 2, activation="relu",
              kernel_regularizer=l2(l2_reg), name="head_dense_2")(x)
    x = Dropout(dropout * 0.5)(x)
    output = Dense(num_classes, activation="softmax", name="stage_output")(x)

    model = Model(inputs=inp, outputs=output, name="SleepNet_CNN_GRU_LSTM")

    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ────────────────────────────────────────────────────────────────
# Model from config dict (for MOPSO trial flexibility)
# ────────────────────────────────────────────────────────────────

def build_from_config(cfg: dict, window_size: int, n_features: int) -> Model:
    """Convenience wrapper used by the MOPSO fitness evaluator."""
    return build_model(
        window_size=window_size,
        n_features=n_features,
        num_classes=cfg.get("num_classes", 4),
        cnn_filters_1=int(cfg.get("cnn_filters_1", 64)),
        cnn_filters_2=int(cfg.get("cnn_filters_2", 128)),
        cnn_kernel_1=int(cfg.get("cnn_kernel_1", 3)),
        cnn_kernel_2=int(cfg.get("cnn_kernel_2", 3)),
        gru_units=int(cfg.get("gru_units", 64)),
        lstm_units=int(cfg.get("lstm_units", 64)),
        dense_units=int(cfg.get("dense_units", 128)),
        dropout=float(cfg.get("dropout", 0.30)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )


# ────────────────────────────────────────────────────────────────
# Model summary helper
# ────────────────────────────────────────────────────────────────

def model_param_count(model: Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def print_model_summary(model: Model) -> None:
    model.summary(line_length=100)
    print(f"\nTrainable parameters: {model_param_count(model):,}")
