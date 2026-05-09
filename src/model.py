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
# Focal Loss — critical for class-imbalanced sleep stage detection
# ────────────────────────────────────────────────────────────────

class SparseFocalLoss(tf.keras.losses.Loss):
    """Sparse focal loss for multi-class classification.

    Focal loss down-weights easy examples and focuses training on
    hard-to-classify minority classes (REM, Deep).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, class_weights: dict | None = None,
                 name: str = "sparse_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.class_weights = class_weights  # {0: w0, 1: w1, ...}

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_one_hot = tf.one_hot(y_true, num_classes)

        p_t = tf.reduce_sum(y_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        ce = -tf.reduce_sum(y_one_hot * tf.math.log(y_pred), axis=-1)

        loss = focal_weight * ce

        # Apply per-class alpha weights
        if self.class_weights is not None:
            n_classes = len(self.class_weights)
            alpha = tf.constant([self.class_weights.get(i, 1.0)
                                 for i in range(n_classes)], dtype=tf.float32)
            alpha_t = tf.reduce_sum(y_one_hot * alpha, axis=-1)
            loss = alpha_t * loss

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config


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


def _squeeze_excite(x: tf.Tensor, ratio: int = 4) -> tf.Tensor:
    """Squeeze-and-Excitation channel attention block.

    Learns to re-weight feature channels — helps the model attend to
    the most discriminative signals for each sleep stage.
    """
    filters = x.shape[-1]
    se = GlobalAveragePooling1D()(x)                       # [B, C]
    se = Dense(max(filters // ratio, 4), activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)           # [B, C]
    se = Reshape((1, filters))(se)                          # [B, 1, C]
    return x * se                                           # broadcast multiply


# ────────────────────────────────────────────────────────────────
# Main model builder
# ────────────────────────────────────────────────────────────────

def build_model(
    window_size: int = 30,
    n_features: int = 22,
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
    focal_gamma: float = 2.0,
    class_weights: dict | None = None,
    use_cnn: bool = True,
    use_gru: bool = True,
    use_lstm: bool = True,
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
    if not any([use_cnn, use_gru, use_lstm]):
        raise ValueError("At least one component must be enabled: CNN, GRU, or LSTM.")

    inp = Input(shape=(window_size, n_features), name="input_sequence")

    # ── CNN encoder ────────────────────────────────────────────
    if use_cnn:
        cnn = _conv_block(inp, cnn_filters_1, cnn_kernel_1, pool=False, l2_reg=l2_reg)
        cnn = _residual_conv_block(cnn, cnn_filters_1, cnn_kernel_1, l2_reg=l2_reg)
        cnn = _squeeze_excite(cnn)      # channel attention after first residual
        cnn = _conv_block(cnn, cnn_filters_2, cnn_kernel_2, pool=True,  l2_reg=l2_reg)
        cnn = _residual_conv_block(cnn, cnn_filters_2, cnn_kernel_2, l2_reg=l2_reg)
        cnn = _squeeze_excite(cnn)      # channel attention after second residual
        sequence_features = cnn
    else:
        sequence_features = inp

    pooled_branches = []

    # ── GRU branch (short-term disturbances) ───────────────────
    if use_gru:
        gru_out = Bidirectional(
            GRU(gru_units, return_sequences=True, dropout=0.1,
                recurrent_dropout=0.0,
                kernel_regularizer=l2(l2_reg)),
            name="bidirectional_gru"
        )(sequence_features)
        gru_out = LayerNormalization()(gru_out)
        pooled_branches.append(GlobalAveragePooling1D(name="gru_pool")(gru_out))

    # ── LSTM branch (long-term sleep cycles) ───────────────────
    if use_lstm:
        lstm_out = Bidirectional(
            LSTM(lstm_units, return_sequences=True, dropout=0.1,
                 recurrent_dropout=0.0,
                 kernel_regularizer=l2(l2_reg)),
            name="bidirectional_lstm"
        )(sequence_features)
        lstm_out = LayerNormalization()(lstm_out)
        pooled_branches.append(GlobalAveragePooling1D(name="lstm_pool")(lstm_out))

    # ── CNN global context ─────────────────────────────────────
    if use_cnn:
        pooled_branches.insert(0, GlobalAveragePooling1D(name="cnn_pool")(cnn))
    elif not pooled_branches:
        pooled_branches.append(GlobalAveragePooling1D(name="raw_pool")(inp))

    # ── Merge all branches ─────────────────────────────────────
    merged = pooled_branches[0] if len(pooled_branches) == 1 else Concatenate(name="merge")(pooled_branches)
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

    enabled = []
    if use_cnn:
        enabled.append("CNN")
    if use_gru:
        enabled.append("GRU")
    if use_lstm:
        enabled.append("LSTM")
    model = Model(inputs=inp, outputs=output, name="SleepNet_" + "_".join(enabled))

    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

    # Use focal loss to handle class imbalance (critical for REM/Deep)
    loss_fn = SparseFocalLoss(gamma=focal_gamma, class_weights=class_weights)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


# ────────────────────────────────────────────────────────────────
# Model from config dict (for MOPSO trial flexibility)
# ────────────────────────────────────────────────────────────────

def build_from_config(cfg: dict, window_size: int, n_features: int,
                      class_weights: dict | None = None) -> Model:
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
        class_weights=class_weights,
        use_cnn=bool(cfg.get("use_cnn", True)),
        use_gru=bool(cfg.get("use_gru", True)),
        use_lstm=bool(cfg.get("use_lstm", True)),
    )


# ────────────────────────────────────────────────────────────────
# Model summary helper
# ────────────────────────────────────────────────────────────────

def model_param_count(model: Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def print_model_summary(model: Model) -> None:
    model.summary(line_length=100)
    print(f"\nTrainable parameters: {model_param_count(model):,}")
