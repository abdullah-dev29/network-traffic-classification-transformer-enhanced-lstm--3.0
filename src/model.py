"""
Transformer-Enhanced LSTM for binary Darknet-vs-Benign classification.

This model evolves the Phase 1 CNN-LSTM baseline rather than replacing it.
It keeps the CNN front-end (local feature extraction) and an LSTM
(sequential modeling), then -- instead of compressing the sequence with a
second LSTM -- inserts a Transformer encoder block that applies multi-head
self-attention across all positions of the LSTM's output sequence. This
lets the model weigh long-range relationships between flow features and
focus on the most informative parts of the representation, before a
global-pooling + dense head produces the binary decision. As in the
baseline, there is no Embedding layer (input is tabular continuous
features) and the LSTM is a standard Keras LSTM.

Built with the Keras Functional API (required for the transformer block's
residual connections). This module never fits a model -- it only builds
and (optionally, under __main__) prints model.summary().
"""

import os
import sys

# repo root is the parent of src/, needed since this is run as
# `python src/model.py` (only src/ is auto-added to sys.path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam


def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout):
    """Post-norm Transformer encoder block: self-attention then feed-forward, each with residual + LayerNorm."""
    # --- Multi-Head Self-Attention sublayer (residual + LayerNorm) ---
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout
    )(inputs, inputs)  # self-attention: query=key=value
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attn)  # residual add + norm

    # --- Position-wise Feed-Forward sublayer (residual + LayerNorm) ---
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(inputs.shape[-1])(ff)  # project back to embed_dim
    ff = layers.Dropout(dropout)(ff)
    out = layers.LayerNormalization(epsilon=1e-6)(x + ff)  # residual add + norm
    return out


class LearnablePositionalEncoding(layers.Layer):
    """Simple learnable positional encoding, added before the first transformer block.

    Only used when config.USE_POSITIONAL_ENCODING is True. The LSTM already
    encodes order, so this is an optional toggle for experimentation.
    """

    def build(self, input_shape):
        seq_len, embed_dim = input_shape[1], input_shape[2]
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(seq_len, embed_dim),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.pos_embedding


def build_transformer_lstm(n_features: int, n_channels: int = 1) -> tf.keras.Model:
    """Build and compile the Transformer-Enhanced LSTM. Does not fit the model."""
    inputs = Input(shape=(n_features, n_channels))

    x = layers.Conv1D(
        config.CONV1_FILTERS, config.CONV1_KERNEL, activation="relu", padding=config.CONV_PADDING
    )(inputs)
    x = layers.MaxPooling1D(pool_size=config.POOL_SIZE)(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    x = layers.Conv1D(
        config.CONV2_FILTERS, config.CONV2_KERNEL, activation="relu", padding=config.CONV_PADDING
    )(x)
    x = layers.MaxPooling1D(pool_size=config.POOL_SIZE)(x)

    x = layers.LSTM(config.LSTM_UNITS, return_sequences=True)(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    if config.USE_POSITIONAL_ENCODING:
        x = LearnablePositionalEncoding()(x)

    for _ in range(config.NUM_TRANSFORMER_BLOCKS):
        x = transformer_encoder(
            x,
            num_heads=config.NUM_HEADS,
            key_dim=config.KEY_DIM,
            ff_dim=config.FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
        )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(config.DENSE_HEAD_UNITS, activation="relu")(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="transformer_enhanced_lstm")

    if config.LOSS_FN == "focal":
        loss = BinaryFocalCrossentropy()
    else:
        loss = "binary_crossentropy"

    model.compile(
        optimizer=Adam(),
        loss=loss,
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )
    return model


if __name__ == "__main__":
    # Shape self-verification only -- builds and prints summary, never fits.
    demo_model = build_transformer_lstm(n_features=62)
    demo_model.summary()
