import gradio as gr
import tensorflow as tf
import numpy as np
import pickle
import string
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# 1. RECREATE CUSTOM LAYERS (exact same code)
# ============================================


@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embed_dim
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=self.sequence_length, output_dim=self.embed_dim
        )
        super().build(input_shape)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.dense_dim, activation="gelu"),
                tf.keras.layers.Dense(self.dense_dim, activation="gelu"),
                tf.keras.layers.Dense(self.dense_dim, activation="gelu"),
                tf.keras.layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        super().build(input_shape)

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.dense_dim, activation="gelu"),
                tf.keras.layers.Dense(self.dense_dim, activation="gelu"),
                tf.keras.layers.Dense(self.dense_dim, activation="gelu"),
                tf.keras.layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        super().build(input_shape)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="float32")
            combined_mask = tf.minimum(padding_mask, causal_mask)
        else:
            combined_mask = causal_mask

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=combined_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=mask
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_length = input_shape[0], input_shape[1]
        i = tf.range(seq_length)[:, tf.newaxis]
        j = tf.range(seq_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, seq_length, seq_length))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


# ============================================
# 1b. CUSTOM LEARNING RATE SCHEDULE
# Required for loading the enhanced model
# ============================================


@tf.keras.utils.register_keras_serializable(package="Custom")
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule with warmup"""

    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": int(self.d_model.numpy()), "warmup_steps": self.warmup_steps}


# ============================================
# 2. LOAD MODEL AND PREPROCESSOR
# ============================================

logger.info("Loading enhanced transformer model and vocabularies...")

# Load config (only needs vocab_size and sequence_length for preprocessing)
with open("models/config.pkl", "rb") as f:
    config = pickle.load(f)

# Load vocabularies
with open("models/source_vocab.pkl", "rb") as f:
    source_vocab = pickle.load(f)

with open("models/target_vocab.pkl", "rb") as f:
    target_vocab = pickle.load(f)

# Recreate vectorization layers (this is what needs the config)
strip_chars = string.punctuation.replace("[", "").replace("]", "")


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


source_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=config["vocab_size"],
    output_mode="int",
    output_sequence_length=config["sequence_length"],
)
source_vectorization.set_vocabulary(source_vocab)

target_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=config["vocab_size"],
    output_mode="int",
    output_sequence_length=config["sequence_length"] + 1,
    standardize=custom_standardization,
)
target_vectorization.set_vocabulary(target_vocab)

# Load ENHANCED model (has its own architecture baked in)
model = tf.keras.models.load_model(
    "models/enhanced_transformer.keras",
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder,
        "CustomSchedule": CustomSchedule,  # Required for enhanced model
    },
)

logger.info("Enhanced transformer model loaded successfully!")
logger.info(f"Model parameters: {model.count_params():,}")

# ============================================
# 3. TRANSLATION FUNCTION
# ============================================


def translate_text(english_text, use_beam_search=True):
    """Translate English to French"""

    if not english_text.strip():
        return "Please enter some text to translate."

    # Get French vocabulary
    fr_vocab = target_vectorization.get_vocabulary()
    fr_index_lookup = dict(zip(range(len(fr_vocab)), fr_vocab))

    # Tokenize input
    tokenized_input = source_vectorization([english_text])

    if use_beam_search:
        # Beam search (better quality)
        return beam_search_decode(tokenized_input, fr_vocab, fr_index_lookup)
    else:
        # Greedy decoding (faster)
        return greedy_decode(tokenized_input, fr_vocab, fr_index_lookup)


def greedy_decode(tokenized_input, fr_vocab, fr_index_lookup):
    """Greedy decoding"""
    decoded_sentence = "[start]"

    for i in range(20):
        tokenized_target = target_vectorization([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input, tokenized_target])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = fr_index_lookup[sampled_token_index]

        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break

    return decoded_sentence.replace("[start] ", "").replace(" [end]", "")


def beam_search_decode(tokenized_input, fr_vocab, fr_index_lookup, beam_width=3):
    """Beam search decoding (better quality)"""
    import heapq

    start_token_idx = fr_vocab.index("[start]")
    end_token_idx = fr_vocab.index("[end]")

    beams = [(0.0, [start_token_idx], False)]

    for _ in range(20):
        candidates = []

        for score, sequence, finished in beams:
            if finished:
                candidates.append((score, sequence, True))
                continue

            decoded_so_far = " ".join([fr_index_lookup[idx] for idx in sequence])
            tokenized_target = target_vectorization([decoded_so_far])[:, :-1]

            predictions = model([tokenized_input, tokenized_target])
            next_token_probs = predictions[0, len(sequence) - 1, :]

            top_k_indices = tf.math.top_k(
                next_token_probs, k=beam_width
            ).indices.numpy()
            top_k_probs = tf.math.top_k(next_token_probs, k=beam_width).values.numpy()

            for token_idx, prob in zip(top_k_indices, top_k_probs):
                new_score = score - np.log(prob + 1e-10)
                new_sequence = sequence + [int(token_idx)]
                is_finished = token_idx == end_token_idx
                candidates.append((new_score, new_sequence, is_finished))

        beams = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

        if all(finished for _, _, finished in beams):
            break

    best_score, best_sequence, _ = min(beams, key=lambda x: x[0])

    best_translation = " ".join(
        [
            fr_index_lookup[idx]
            for idx in best_sequence
            if idx not in [0, start_token_idx, end_token_idx]
        ]
    )

    return best_translation


# ============================================
# 4. GRADIO INTERFACE
# ============================================

demo = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(
            label="English Text",
            placeholder="Enter English text to translate...",
            lines=3,
        ),
        gr.Checkbox(label="Use Beam Search (better quality, slower)", value=True),
    ],
    outputs=gr.Textbox(label="French Translation", lines=3),
    title="English to French Enhanced Transformer",
    description="""
    **Enhanced Transformer** architecture built from scratch in Keras/TensorFlow.

    *Built by Jeremy Vachier*

    **Architecture:**
    - 2 encoder layers, 2 decoder layers
    - 128-dimensional embeddings
    - 4 attention heads per layer
    - 2048-dimensional feed-forward networks
    - Custom learning rate schedule with warmup

    **Performance:**
    - 44,967,200 parameters
    - BLEU score: 0.58
    - No pre-trained models used - built from scratch!

    **Try it:** Enter English text and get French translation!
    - Beam Search: Higher quality, slightly slower
    - Greedy: Faster, good quality

    [View on Kaggle](https://www.kaggle.com/code/jvachier/transformer-nmt-en-fr) |
    [GitHub](https://github.com/jvachier/Sentiment_Analysis)
    """,
)

if __name__ == "__main__":
    demo.launch()
