import tensorflow as tf
import numpy as np


def positional_encoding(max_length, embed_size):
  p, i = np.meshgrid(np.arange(max_length), 2*np.arange(embed_size//2))
  pos_emb = np.empty((1, max_length, embed_size))

  pos_emb[0, :, ::2] = np.sin(p/10_000 ** (i/embed_size)).T
  pos_emb[0, :, 1::2] = np.cos(p/10_000 ** (i/embed_size)).T

  return tf.constant(tf.cast(pos_emb, tf.float32))

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embed_size):
    super().__init__()
    self.embed_size = embed_size
    self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
    self.pos_encoding = positional_encoding(100, embed_size)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    x = self.embedding(x)

    x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
    x = x + self.pos_encoding
    return x


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(query=x,
                                        key=context,
                                        value=context,
                                        return_attention_scores=True)
    self.last_attn_scores = attn_scores
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class GlobalAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(query=x, value=x)

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

class CausalMaskAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(query=x, value=x, use_causal_mask=True)

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, embed_size, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"),
        tf.keras.layers.Dense(embed_size),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layernorm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    seq_out = self.seq(x)
    x = self.add([x, seq_out])
    x = self.layernorm(x)
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, embed_size, num_heads, dff, dropout_rate=0.1):
    super().__init__()
    self.self_attention = GlobalAttention(num_heads=num_heads,
                                          key_dim=embed_size,
                                          dropout=dropout_rate)

    self.ffn = FeedForward(embed_size, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)

    return x

class Encoder(tf.keras.Model):
  def __init__(self, num_layers, embed_size, num_heads, dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.num_layers = num_layers
    self.embed_size = embed_size
    self.encoder_layer = EncoderLayer(embed_size, num_heads, dff)
    self.pos_embed = PositionalEmbedding(vocab_size, embed_size)
    self.enc_layer = [self.encoder_layer for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    x = self.pos_embed(x)
    x = self.dropout(x)
    for i in range(self.num_layers):
       x = self.enc_layer[i](x)
    return x

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, embed_size, num_heads, dff, dropout_rate=0.1):
    super().__init__()
    self.causal_mask_attn = CausalMaskAttention(num_heads=num_heads,
                                                key_dim=embed_size,
                                                dropout=dropout_rate)
    self.cross_attn = CrossAttention(num_heads=num_heads,
                                     key_dim=embed_size,
                                     dropout=dropout_rate)
    self.ffn = FeedForward(embed_size, dff, dropout_rate=dropout_rate)

  def call(self, x, context):
    x = self.causal_mask_attn(x=x)
    x = self.cross_attn(x=x, context=context)

    self.last_attn_scores = self.cross_attn.last_attn_scores

    x = self.ffn(x)
    return x

class Decoder(tf.keras.Model):
  def __init__(self, num_layers, embed_size, num_heads, dff, vocab_size, dropout_rate=0.1):
    super().__init__()
    self.num_layers = num_layers
    self.embed_size = embed_size
    self.decoder_layer = DecoderLayer(embed_size, num_heads, dff)
    self.pos_embed = PositionalEmbedding(vocab_size, embed_size)
    self.dec_layer = [self.decoder_layer for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    self.last_attn_scores = None

  def call(self, x, context):
    x = self.pos_embed(x)
    x = self.dropout(x)
    for i in range(self.num_layers):
       x = self.dec_layer[i](x, context)

    self.last_attn_scores = self.dec_layer[-1].last_attn_scores
    return x

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, embed_size, num_heads, dff,
               vocab_size, dropout_rate=0.1):
    super().__init__()
    self.num_layers = num_layers
    self.embed_size = embed_size

    self.encoder = Encoder(num_layers, embed_size, num_heads, dff, vocab_size)
    self.decoder = Decoder(num_layers, embed_size, num_heads, dff, vocab_size)
    self.final_layer = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs):
    enc_input, dec_input = inputs

    enc_output = self.encoder(enc_input)
    dec_output = self.decoder(dec_input, enc_output)

    logits = self.final_layer(dec_output)


    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, embed_size, warmup_steps=4000):
    super().__init__()

    self.embed_size  = tf.cast(embed_size, dtype=tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.embed_size) * tf.math.minimum(arg1, arg2)
  def get_config(self):
    return {"embed_size": self.embed_size.numpy(), "warmup_steps": self.warmup_steps}