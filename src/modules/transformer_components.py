import tensorflow as tf
import logging
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from typing import Any


@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        """
        Initialize the PositionalEmbedding layer.

        Args:
            sequence_length (int): Maximum sequence length.
            vocab_size (int): Vocabulary size.
            embed_dim (int): Embedding dimension.
            kwargs: Additional keyword arguments for the parent class.
        """
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


def evaluate_bleu(
    model: tf.keras.Model, dataset: tf.data.Dataset, preprocessor: Any
) -> float:
    """
    Evaluate the BLEU score for the model on the given dataset.

    Args:
        model (tf.keras.Model): The trained Transformer model.
        dataset (tf.data.Dataset): The dataset to evaluate.
        preprocessor (TextPreprocessor): The text preprocessor for decoding.

    Returns:
        float: The BLEU score for the dataset.
    """
    logging.info("Starting BLEU score evaluation.")
    references = []
    candidates = []
    smoothing_function = SmoothingFunction().method1

    # Get the vocabulary from the target vectorization layer
    vocab = preprocessor.target_vectorization.get_vocabulary()
    index_to_word = {i: word for i, word in enumerate(vocab)}

    for batch in dataset:
        inputs, targets = batch
        # Generate predictions
        predictions = model.predict(inputs, verbose=0)

        # Decode predictions and targets
        for i in range(len(predictions)):
            # Decode predicted sentence
            pred_tokens = predictions[i].argmax(axis=-1)  # Get token IDs
            pred_sentence = " ".join(
                [
                    index_to_word[token] for token in pred_tokens if token != 0
                ]  # Ignore padding tokens
            )

            # Decode reference sentence
            ref_tokens = targets[i].numpy()  # Get token IDs
            ref_sentence = " ".join(
                [
                    index_to_word[token] for token in ref_tokens if token != 0
                ]  # Ignore padding tokens
            )

            candidates.append(pred_sentence)
            references.append([ref_sentence])

    # Calculate BLEU score
    bleu_score = corpus_bleu(
        references, candidates, smoothing_function=smoothing_function
    )
    logging.info(f"BLEU score evaluation completed: {bleu_score:.4f}")
    return bleu_score
