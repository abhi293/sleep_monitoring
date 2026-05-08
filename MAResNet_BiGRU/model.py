from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GRU,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
    Multiply,
    Reshape,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: dict | None = None,
        label_smoothing: float = 0.0,
        name: str = "sparse_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        n_classes = tf.shape(y_pred)[-1]
        y_one_hot = tf.one_hot(y_true, n_classes)
        if self.label_smoothing:
            smooth = tf.cast(self.label_smoothing, y_pred.dtype)
            y_one_hot = y_one_hot * (1.0 - smooth) + smooth / tf.cast(n_classes, y_pred.dtype)

        pt = tf.reduce_sum(y_one_hot * y_pred, axis=-1)
        ce = -tf.reduce_sum(y_one_hot * tf.math.log(y_pred), axis=-1)
        loss = tf.pow(1.0 - pt, self.gamma) * ce
        if self.class_weights is not None:
            alpha = tf.constant(
                [self.class_weights.get(i, 1.0) for i in range(len(self.class_weights))],
                dtype=tf.float32,
            )
            hard_labels = tf.one_hot(y_true, len(self.class_weights))
            loss = loss * tf.reduce_sum(hard_labels * alpha, axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "label_smoothing": self.label_smoothing})
        return cfg


def _conv_bn_relu(x, filters: int, kernel: int, dilation: int = 1, l2_reg: float = 1e-4):
    x = Conv1D(
        filters,
        kernel_size=kernel,
        padding="same",
        dilation_rate=dilation,
        kernel_regularizer=l2(l2_reg),
    )(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def _channel_attention(x, ratio: int = 8):
    channels = int(x.shape[-1])
    pooled = GlobalAveragePooling1D()(x)
    gate = Dense(max(channels // ratio, 8), activation="relu")(pooled)
    gate = Dense(channels, activation="sigmoid")(gate)
    gate = Reshape((1, channels))(gate)
    return Multiply()([x, gate])


def _multi_scale_residual_block(
    x,
    filters: int,
    kernels: Iterable[int],
    dilation_rates: Iterable[int],
    dropout: float,
    l2_reg: float,
    name: str,
):
    branches = []
    for i, (kernel, dilation) in enumerate(zip(kernels, dilation_rates)):
        branch = _conv_bn_relu(x, filters, kernel, dilation, l2_reg)
        branch = _conv_bn_relu(branch, filters, 1, 1, l2_reg)
        branches.append(branch)

    merged = Concatenate(name=f"{name}_multi_scale_concat")(branches)
    merged = Conv1D(filters, kernel_size=1, padding="same", kernel_regularizer=l2(l2_reg))(merged)
    merged = BatchNormalization()(merged)
    merged = _channel_attention(merged)
    merged = Dropout(dropout * 0.5)(merged)

    shortcut = x
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same", kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    out = Add(name=f"{name}_residual_add")([shortcut, merged])
    return Activation("relu", name=f"{name}_out")(out)


def _temporal_attention(x, heads: int, key_dim: int, dropout: float):
    norm = LayerNormalization()(x)
    attn = MultiHeadAttention(num_heads=heads, key_dim=key_dim, dropout=dropout)(norm, norm)
    x = Add()([x, attn])
    return LayerNormalization()(x)


def build_model(
    window_size: int = 30,
    n_features: int = 28,
    num_classes: int = 4,
    base_filters: int = 32,
    resnet_blocks: int = 2,
    kernels: list[int] | tuple[int, ...] = (3, 5, 7),
    dilation_rates: list[int] | tuple[int, ...] = (1, 2, 3),
    attention_heads: int = 2,
    attention_key_dim: int = 16,
    bigru_units: int = 48,
    bigru_layers: int = 1,
    dense_units: int = 96,
    dropout: float = 0.45,
    learning_rate: float = 5e-4,
    l2_reg: float = 5e-4,
    focal_gamma: float = 1.5,
    label_smoothing: float = 0.05,
    class_weights: dict | None = None,
) -> Model:
    inputs = Input(shape=(window_size, n_features), name="sensor_window")

    # Sensor gate lets the model weight HR, GSR, audio, movement, and environment channels per window.
    sensor_gate = Dense(n_features, activation="sigmoid", name="sensor_attention_gate")(inputs)
    x = Multiply(name="sensor_attention")([inputs, sensor_gate])
    x = _conv_bn_relu(x, base_filters, 1, 1, l2_reg)

    for block_idx in range(resnet_blocks):
        x = _multi_scale_residual_block(
            x,
            filters=base_filters,
            kernels=kernels,
            dilation_rates=dilation_rates,
            dropout=dropout,
            l2_reg=l2_reg,
            name=f"mares_block_{block_idx + 1}",
        )
        if block_idx == 0:
            x = MaxPooling1D(pool_size=2, padding="same", name="early_temporal_pool")(x)

    x = _temporal_attention(x, attention_heads, attention_key_dim, dropout * 0.5)

    seq = x
    for layer_idx in range(bigru_layers):
        seq = Bidirectional(
            GRU(
                bigru_units,
                return_sequences=True,
                dropout=dropout * 0.35,
                recurrent_dropout=0.0,
                kernel_regularizer=l2(l2_reg),
            ),
            name=f"bidirectional_gru_{layer_idx + 1}",
        )(seq)
        seq = LayerNormalization(name=f"bigru_norm_{layer_idx + 1}")(seq)

    context = GlobalAveragePooling1D(name="temporal_context_pool")(seq)
    cnn_context = GlobalAveragePooling1D(name="maresnet_context_pool")(x)
    merged = Concatenate(name="context_fusion")([context, cnn_context])

    z = Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg), name="head_dense_1")(merged)
    z = BatchNormalization()(z)
    z = Dropout(dropout, name="head_dropout_1")(z)
    z = Dense(dense_units // 2, activation="relu", kernel_regularizer=l2(l2_reg), name="head_dense_2")(z)
    z = Dropout(dropout * 0.5, name="head_dropout_2")(z)
    outputs = Dense(num_classes, activation="softmax", name="stage_output")(z)

    model = Model(inputs, outputs, name="MAResNet_BiGRU_SleepNet")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=SparseFocalLoss(
            gamma=focal_gamma,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        ),
        metrics=["accuracy"],
    )
    return model


def model_param_count(model: Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def print_model_summary(model: Model) -> None:
    model.summary(line_length=120)
    print(f"\nTrainable parameters: {model_param_count(model):,}")
