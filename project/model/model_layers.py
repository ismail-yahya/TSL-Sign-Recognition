"""
Custom layers and loss from the Transformer training (03_TID_Model_Training).
Required for loading best_model_transformer.keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SparseCategoricalCrossentropyWithLS(keras.losses.Loss):
    """Sparse Categorical Crossentropy with Label Smoothing."""

    def __init__(self, num_classes, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, self.num_classes)
        y_true_smooth = y_true_one_hot * (1.0 - self.label_smoothing) + (
            self.label_smoothing / tf.cast(self.num_classes, tf.float32)
        )
        return tf.reduce_mean(keras.losses.categorical_crossentropy(y_true_smooth, y_pred))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "label_smoothing": self.label_smoothing})
        return config


class TransformerBlock(layers.Layer):
    """Transformer Encoder block: Multi-Head Attention + FFN + LayerNorm."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEmbedding(layers.Layer):
    """Learned positional embedding for sequence positions."""

    def __init__(self, maxlen, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
