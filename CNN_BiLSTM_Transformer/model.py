from __future__ import annotations

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
    LSTM,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
    SpatialDropout1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        gamma: float = 1.5,
        class_weights: dict | None = None,
        label_smoothing: float = 0.05,
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
            hard = tf.one_hot(y_true, len(self.class_weights))
            loss = loss * tf.reduce_sum(hard * alpha, axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "label_smoothing": self.label_smoothing})
        return cfg


class SparseSmoothedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, label_smoothing: float = 0.02, name: str = "sparse_smoothed_cross_entropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        n_classes = tf.shape(y_pred)[-1]
        y_one_hot = tf.one_hot(y_true, n_classes)
        if self.label_smoothing:
            smooth = tf.cast(self.label_smoothing, y_pred.dtype)
            y_one_hot = y_one_hot * (1.0 - smooth) + smooth / tf.cast(n_classes, y_pred.dtype)
        return tf.reduce_mean(-tf.reduce_sum(y_one_hot * tf.math.log(y_pred), axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"label_smoothing": self.label_smoothing})
        return cfg


def _conv_block(x, filters: int, kernel: int, dropout: float, l2_reg: float, pool: bool = False):
    shortcut = x
    x = Conv1D(filters, kernel, padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SpatialDropout1D(dropout * 0.5)(x)
    x = Conv1D(filters, kernel, padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv1D(filters, 1, padding="same", kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    if pool:
        x = MaxPooling1D(pool_size=2, padding="same")(x)
    return x


def _transformer_encoder(x, heads: int, key_dim: int, ff_dim: int, dropout: float, l2_reg: float, name: str):
    attn_in = LayerNormalization(name=f"{name}_attn_norm")(x)
    attn = MultiHeadAttention(num_heads=heads, key_dim=key_dim, dropout=dropout, name=f"{name}_mha")(attn_in, attn_in)
    x = Add(name=f"{name}_attn_add")([x, attn])

    ff_in = LayerNormalization(name=f"{name}_ff_norm")(x)
    ff = Dense(ff_dim, activation="relu", kernel_regularizer=l2(l2_reg), name=f"{name}_ff_1")(ff_in)
    ff = Dropout(dropout, name=f"{name}_ff_dropout")(ff)
    ff = Dense(int(x.shape[-1]), kernel_regularizer=l2(l2_reg), name=f"{name}_ff_2")(ff)
    return Add(name=f"{name}_ff_add")([x, ff])


def build_model(
    window_size: int = 30,
    n_features: int = 28,
    num_classes: int = 4,
    use_bilstm: bool = True,
    cnn_filters: int = 48,
    lstm_units: int = 48,
    transformer_blocks: int = 2,
    attention_heads: int = 3,
    attention_key_dim: int = 24,
    transformer_ff_dim: int = 128,
    dense_units: int = 96,
    dropout: float = 0.40,
    learning_rate: float = 5e-4,
    l2_reg: float = 5e-4,
    focal_gamma: float = 1.5,
    label_smoothing: float = 0.05,
    class_weights: dict | None = None,
    loss_type: str = "ce",
) -> Model:
    inputs = Input(shape=(window_size, n_features), name="sensor_window")

    x = Conv1D(cnn_filters, 3, padding="same", kernel_regularizer=l2(l2_reg), name="cnn_stem")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = _conv_block(x, cnn_filters, 3, dropout, l2_reg, pool=False)
    x = _conv_block(x, cnn_filters * 2, 5, dropout, l2_reg, pool=True)

    if use_bilstm:
        x = Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=True,
                dropout=dropout * 0.35,
                recurrent_dropout=0.0,
                kernel_regularizer=l2(l2_reg),
            ),
            name="bidirectional_lstm",
        )(x)
        x = LayerNormalization(name="bilstm_norm")(x)

    for i in range(transformer_blocks):
        x = _transformer_encoder(
            x,
            heads=attention_heads,
            key_dim=attention_key_dim,
            ff_dim=transformer_ff_dim,
            dropout=dropout * 0.5,
            l2_reg=l2_reg,
            name=f"transformer_{i + 1}",
        )

    pooled = GlobalAveragePooling1D(name="sequence_pool")(x)
    head = Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg), name="head_dense_1")(pooled)
    head = BatchNormalization()(head)
    head = Dropout(dropout, name="head_dropout_1")(head)
    head = Dense(dense_units // 2, activation="relu", kernel_regularizer=l2(l2_reg), name="head_dense_2")(head)
    head = Dropout(dropout * 0.5, name="head_dropout_2")(head)
    outputs = Dense(num_classes, activation="softmax", name="stage_output")(head)

    name = "CNN_BiLSTM_Transformer_SleepNet" if use_bilstm else "CNN_Transformer_SleepNet"
    model = Model(inputs, outputs, name=name)
    if loss_type == "focal":
        loss_fn = SparseFocalLoss(
            gamma=focal_gamma,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        loss_fn = SparseSmoothedCrossEntropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


def model_param_count(model: Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def print_model_summary(model: Model) -> None:
    model.summary(line_length=120)
    print(f"\nTrainable parameters: {model_param_count(model):,}")
