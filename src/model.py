"""
Transformer-Enhanced LSTM, shared between two tasks:

- "binary": Darknet vs Benign (sigmoid head, binary_crossentropy / focal).
- "application": 8-class application type from Label.1 (softmax head,
  sparse_categorical_crossentropy).

This model evolves the Phase 1 CNN-LSTM baseline rather than replacing it.
It keeps the CNN front-end (local feature extraction) and an LSTM
(sequential modeling, optionally Bidirectional), then -- instead of
compressing the sequence with a second LSTM -- stacks Transformer encoder
blocks that apply multi-head self-attention across all positions of the
LSTM's output sequence. This lets the model weigh long-range relationships
between flow features and focus on the most informative parts of the
representation, before a global-pooling + dense head produces the
decision. As in the baseline, there is no Embedding layer (input is
tabular continuous features) and the LSTM is a standard Keras LSTM.

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
    ff = layers.Dense(inputs.shape[-1])(ff)  # project back to embed_dim (adapts to BiLSTM's doubled width too)
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


def build_transformer_lstm(
    n_features: int, n_classes: int, task: str, n_channels: int = 1
) -> tf.keras.Model:
    """Build and compile the Transformer-Enhanced LSTM for `task`. Does not fit the model.

    task == "binary": n_classes is ignored for the head (single sigmoid unit).
    task == "application": n_classes sets the softmax head's width.
    """
    if task not in ("binary", "application"):
        raise ValueError(f"Unknown task: {task!r}")

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

    lstm_layer = layers.LSTM(config.LSTM_UNITS, return_sequences=True)
    if config.USE_BILSTM:
        lstm_layer = layers.Bidirectional(lstm_layer)  # doubles the channel dim (e.g. 100 -> 200)
    x = lstm_layer(x)
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

    if task == "binary":
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = BinaryFocalCrossentropy() if config.LOSS_FN == "focal" else "binary_crossentropy"
        metrics = [
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ]
    else:  # application
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
        # Multiclass precision/recall/AUC are computed in evaluate.py via sklearn instead.
        metrics = ["accuracy"]

    model = Model(inputs=inputs, outputs=outputs, name=f"transformer_enhanced_lstm_{task}")
    model.compile(optimizer=Adam(), loss=loss, metrics=metrics)
    return model


if __name__ == "__main__":
    # Shape self-verification only -- builds and prints summaries, never fits.
    print("=== binary head ===")
    build_transformer_lstm(n_features=62, n_classes=2, task="binary").summary()
    print("\n=== application head ===")
    build_transformer_lstm(n_features=62, n_classes=8, task="application").summary()
