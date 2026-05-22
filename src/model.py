"""
model.py — TSL Sign Recognition: Transformer Architecture Definitions

This module defines the reusable Keras layers that form the Transformer-based
sign language recognition model. Keeping architecture definitions here (separate
from inference logic) follows the Single Responsibility Principle and makes the
model easy to reuse, test, or swap independently.

Architecture overview:
  - TransformerBlock: Single Encoder block (MHSA + FFN + LayerNorm + Dropout)
  - PositionalEmbedding: Learnable positional encoding added to input projections

These classes must be registered with @tf.keras.utils.register_keras_serializable()
so that keras can deserialise them when loading a saved .keras model file.
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """
    Tek bir Transformer Encoder blogu.

    Yapısı:
      1. Multi-Head Self-Attention (MHSA)
      2. Dropout + Residual connection
      3. Layer Normalization
      4. Feed-Forward Network (FFN): Dense(ff_dim, GELU) → Dense(embed_dim)
      5. Dropout + Residual connection
      6. Layer Normalization

    Args:
        embed_dim (int): Boyut sayısı / model genişliği. Varsayılan: 256.
        num_heads (int): Dikkat başı sayısı. Varsayılan: 8.
        ff_dim    (int): FFN ara katman boyutu. Varsayılan: 768.
        rate    (float): Dropout oranı. Varsayılan: 0.1.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 ff_dim: int = 768, rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training: bool = False):
        # --- Self-Attention sub-layer ---
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # --- Feed-Forward sub-layer ---
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "rate":      self.rate,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    """
    Öğrenilebilir Konumsal Gömme (Learnable Positional Embedding) katmanı.

    Transformer mimarisi veriyi paralel işlediğinden dizideki karelerin zamansal
    sırasını doğal olarak bilemez. Bu katman, her konuma özgü bir öğrenilebilir
    vektör ekleyerek modelin hareketin kronolojik akışını kavramasını sağlar.

    Args:
        maxlen    (int): Maksimum dizi uzunluğu (eğitimde kullanılan kare sayısı = 80).
        embed_dim (int): Gömme boyutu — input projection ile aynı olmalı.
    """

    def __init__(self, maxlen: int = 80, embed_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen":    self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config
